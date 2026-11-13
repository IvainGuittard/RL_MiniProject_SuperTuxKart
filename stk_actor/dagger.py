"""DAgger (Ross et al., 2011). Run via learn.py:

    PYTHONPATH=. python -m stk_actor.learn dagger --expert team3
"""

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from .sac import SquashedGaussianActor
from .stk_bridge import expert_action_to_flat


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


def bc_loss(actor, obs, target):
    # MSE on accel/steer, BCE on brake/drift/nitro/rescue (decoded as x > 0)
    mean = actor.mean_layer(actor.backbone(obs))
    return F.mse_loss(torch.tanh(mean[:, :2]), target[:, :2]) + \
        F.binary_cross_entropy_with_logits(mean[:, 2:], (target[:, 2:] > 0).float())


def act(actor, obs):
    with torch.no_grad():
        return actor.act(torch.from_numpy(obs).unsqueeze(0))[0].numpy()


def run_dagger(cfg, env, dataset, expert_cls, writer, evaluator):
    """Train with DAgger; return the evaluator (holds best_actor), like run_rlpd."""
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

    writer.add_scalar("loss/bc", train(cfg.initial_grad_steps), 0)
    env_steps, iteration, done = 0, 0, True
    pbar = tqdm(total=cfg.max_steps)
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
        pbar.update(len(new_obs))

        data_obs.append(torch.from_numpy(np.stack(new_obs)))
        data_act.append(torch.from_numpy(np.stack(new_act)))
        loss = train(cfg.grad_steps_per_iter)
        writer.add_scalar("loss/bc", loss, env_steps)
        writer.add_scalar("dagger/beta", beta, env_steps)
        writer.add_scalar("dagger/dataset_size", sum(len(o) for o in data_obs), env_steps)
        writer.add_scalar("dagger/expert_queries", env_steps, env_steps)

        if result := evaluator.run_if_needed(env_steps, actor):
            done = True  # the evaluation reset the shared env
            pbar.set_description(f"eval={result.mean:7.1f} best={evaluator.best_reward:7.1f}")

    pbar.close()
    return evaluator
