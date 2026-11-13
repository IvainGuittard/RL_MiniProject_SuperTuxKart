"""DAgger (Ross et al., 2011) with the same env, network and expert dataset as RLPD.

    python -m stk_actor.dagger --expert team3 --track lighthouse --seed 1
"""

import argparse
import importlib
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from .sac import SquashedGaussianActor
from .stk_bridge import (
    ACTION_DIM, OBS_DIM, STKVecEnv, collect_expert_dataset, expert_action_to_flat,
    load_dataset, run_expert_lap, save_dataset,
)

ROOT = Path(__file__).resolve().parent.parent
EXPERTS = {
    "team2": ("agents.team2.agent2", "Agent2"),
    "team3": ("agents.team3.agent3", "Agent3"),
    "team4": ("agents.team4.agent4", "Agent4"),
    "team5": ("agents.team5.agent5", "Agent5"),
}


@dataclass(frozen=True)
class DAggerConfig:
    seed: int = 1
    max_steps: int = 300_000
    steps_per_iter: int = 2_000
    beta0: float = 0.5 
    beta_decay: float = 0.5
    initial_grad_steps: int = 5_000
    grad_steps_per_iter: int = 1_000
    batch_size: int = 256
    lr: float = 3e-4
    max_grad_norm: float = 0.5
    actor_hidden: tuple = (256, 256)
    eval_interval: int = 5_000


def resolve_expert(name):
    import pystk2_gymnasium
    sys.path.append(str(ROOT / "expert"))
    module, cls = EXPERTS[name]
    return getattr(importlib.import_module(module), cls)


def bc_loss(actor, obs, target):
    # MSE on accel/steer, BCE on brake/drift/nitro (decoded as x > 0)
    mean = actor.mean_layer(actor.backbone(obs))
    return F.mse_loss(torch.tanh(mean[:, :2]), target[:, :2]) + \
        F.binary_cross_entropy_with_logits(mean[:, 2:], (target[:, 2:] > 0).float())


def act(actor, obs):
    with torch.no_grad():
        return actor.act(torch.from_numpy(obs).unsqueeze(0))[0].numpy()


def evaluate(env, actor):
    """One deterministic lap (the env is deterministic with a single kart)."""
    obs, _ = env.reset()
    steps, done = 0, False
    while not done:
        obs, _, terminated, truncated, info = env.step(act(actor, obs))
        steps += 1
        done = terminated or truncated
    progress = 1.0 if terminated else min(info["distance"] / env.track_length, 1.0)
    return {"finished": terminated, "lap_steps": steps, "progress": progress}


def run_dagger(cfg, env, dataset, expert_cls, run_dir):
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    actor = SquashedGaussianActor(dataset["obs"].shape[1], cfg.actor_hidden, dataset["expert_action"].shape[1])
    optimizer = torch.optim.Adam(actor.parameters(), lr=cfg.lr)
    data_obs, data_act = [dataset["obs"]], [dataset["expert_action"]]

    def train(n_steps):
        obs, target = torch.cat(data_obs), torch.cat(data_act)
        for _ in range(n_steps):
            idx = torch.randint(0, len(obs), (cfg.batch_size,))
            loss = bc_loss(actor, obs[idx], target[idx])
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(), cfg.max_grad_norm)
            optimizer.step()
        return loss.item()

    writer = SummaryWriter(run_dir)

    def evaluate_and_log(env_steps, iteration, loss):
        res = evaluate(env, actor)
        for key, value in res.items():
            writer.add_scalar(f"eval/{key}", float(value), env_steps)
        writer.add_scalar("dagger/expert_queries", env_steps, env_steps)
        writer.add_scalar("dagger/iteration", iteration, env_steps)
        writer.flush()
        print(f"[dagger] env_steps={env_steps} iter={iteration} loss={loss:.4f} {res}", flush=True)

    evaluate_and_log(0, 0, train(cfg.initial_grad_steps))
    env_steps, iteration, last_eval, done = 0, 0, 0, True
    while env_steps < cfg.max_steps:
        iteration += 1
        beta = cfg.beta0 * cfg.beta_decay ** (iteration - 1)
        new_obs, new_act = [], []
        for _ in range(min(cfg.steps_per_iter, cfg.max_steps - env_steps)):
            if done:
                obs, _ = env.reset()
                expert = expert_cls(env)
            label = expert_action_to_flat(expert.choose_action(env.raw_obs))
            new_obs.append(obs)
            new_act.append(label)
            # Mixture policy: expert with probability beta, learner otherwise
            action = label if rng.random() < beta else act(actor, obs)
            obs, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            env_steps += 1

        data_obs.append(torch.from_numpy(np.stack(new_obs)))
        data_act.append(torch.from_numpy(np.stack(new_act)))
        loss = train(cfg.grad_steps_per_iter)
        writer.add_scalar("loss/bc", loss, env_steps)
        writer.add_scalar("dagger/beta", beta, env_steps)
        writer.add_scalar("dagger/dataset_size", sum(len(o) for o in data_obs), env_steps)

        if env_steps - last_eval >= cfg.eval_interval or env_steps >= cfg.max_steps:
            evaluate_and_log(env_steps, iteration, loss)
            last_eval, done = env_steps, True  # the evaluation reset the env

    writer.close()
    torch.save(actor.state_dict(), run_dir / "actor.pt")  # last actor (not the best)
    return actor


def main():
    p = argparse.ArgumentParser(prog="python -m stk_actor.dagger")
    p.add_argument("--expert", default="team3", choices=list(EXPERTS))
    p.add_argument("--track", default="lighthouse")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--max-steps", type=int, default=DAggerConfig.max_steps)
    p.add_argument("--dataset-steps", type=int, default=50_000)
    p.add_argument("--dataset-noise", type=float, default=0.2)
    args = p.parse_args()

    cfg = DAggerConfig(seed=args.seed, max_steps=args.max_steps)
    expert_cls = resolve_expert(args.expert)
    env = STKVecEnv(track=args.track, num_kart=1, seed=args.seed).flat_env
    # Expert dataset, shared with RLPD: collected once, then reloaded
    path = ROOT / "data" / f"expert_{args.expert}_{args.track}_{args.dataset_steps}_noise{args.dataset_noise}.pt"
    if path.exists():
        dataset = load_dataset(path)
    else:
        dataset = collect_expert_dataset(env, expert_cls, args.dataset_steps, args.dataset_noise)
        dataset["meta"].update(expert=args.expert, track=args.track, expert_lap=run_expert_lap(env, expert_cls))
        save_dataset(dataset, path)
    print(f"Dataset {path.name}, expert lap: {dataset['meta']['expert_lap']}")

    run_dir = ROOT / "runs" / "dagger" / args.track / f"seed{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps({**vars(args), **asdict(cfg)}, indent=2))
    run_dagger(cfg, env, dataset, expert_cls, run_dir)


if __name__ == "__main__":
    main()
