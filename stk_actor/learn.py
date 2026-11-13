"""Training entry point.

Modes
-----
sac      — train SAC from scratch on the STK flat env (baseline)
rlpd     — collect expert data, then train RLPD (SAC + offline expert buffer)
bc       — behaviour cloning on the same expert data (baseline for RLPD)
dagger   — DAgger: behaviour cloning + online expert labels (same expert data)
compare  — evaluate saved actors and the expert on the same track suite

All training modes evaluate on the same fixed suite of tracks with a fixed
seed (see stk_eval.py), and save their best actor to <run_dir>/actor.pth.
The expert dataset is cached in data/, so RLPD and BC see the same data.

Run from the repository root:

    PYTHONPATH=. python -m stk_actor.learn rlpd --expert team3 --utd 10
    PYTHONPATH=. python -m stk_actor.learn bc --expert team3
    PYTHONPATH=. python -m stk_actor.learn dagger --expert team3
    PYTHONPATH=. python -m stk_actor.learn compare --expert team3 \\
        --runs runs/rlpd-team3-S1 runs/bc-team3-S1
"""

import argparse
from pathlib import Path


DATA_DIR = Path("data")


def _add_common_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--track", nargs="+", default=None,
                   help="Training track(s), one drawn at each episode (default: all tracks)")
    p.add_argument("--num-kart", type=int, default=1)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--max-steps", type=int, default=300_000)
    p.add_argument("--run-dir", default=None, help="TensorBoard / checkpoint directory")
    _add_eval_args(p)
    p.add_argument("--eval-every", type=int, default=10_000,
                   help="Evaluate every N env steps (BC: gradient steps)")
    p.add_argument("--no-submit-weights", action="store_true",
                   help="Only save <run_dir>/actor.pth, not stk_actor/pystk_actor.pth")


def _add_eval_args(p: argparse.ArgumentParser) -> None:
    from .stk_eval import DEFAULT_EVAL_TRACKS
    p.add_argument("--eval-tracks", nargs="+", default=list(DEFAULT_EVAL_TRACKS))
    p.add_argument("--eval-seed", type=int, default=0)


def _add_expert_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--expert", default="team3",
        choices=["team1", "team2", "team3", "team4", "team5"],
        help="Which expert agent provides the offline data",
    )
    p.add_argument("--offline-steps", type=int, default=50_000,
                   help="Number of expert transitions to collect")


def _evaluator_factory(env, args):
    from .stk_eval import TrackSuiteEvaluator
    return lambda writer: TrackSuiteEvaluator(
        env.flat_env, every=args.eval_every, tracks=args.eval_tracks,
        seed=args.eval_seed, writer=writer,
    )


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def cmd_sac(args: argparse.Namespace) -> None:
    from .sac import SACConfig, run_sac
    from .stk_bridge import STKVecEnv

    env = STKVecEnv(track=args.track, num_kart=args.num_kart, seed=args.seed)

    cfg = SACConfig(
        env_name="stk",        # unused when env is passed directly
        seed=args.seed,
        max_steps=args.max_steps,
        n_envs=1,
        steps_per_round=1,
        updates_per_round=1,
        learning_starts=1_000,
    )

    run_dir = args.run_dir or f"runs/sac-stk-S{args.seed}"
    evaluator = run_sac(cfg, stk_env=env, run_dir=run_dir,
                        make_evaluator=_evaluator_factory(env, args))
    _save_actor(evaluator.best_actor, run_dir, also_submission=not args.no_submit_weights)


def cmd_rlpd(args: argparse.Namespace) -> None:
    from .rlpd import RLPDConfig, run_rlpd
    from .sac import SACConfig
    from .stk_bridge import STKVecEnv, dataset_to_buffer

    # Create the training env first — pystk2 is a per-process singleton, so
    # we reuse its flat_env for expert collection rather than opening two instances.
    env = STKVecEnv(track=args.track, num_kart=args.num_kart, seed=args.seed)
    dataset = _get_dataset(env, args)
    offline_buffer = dataset_to_buffer(dataset)
    print(f"Offline buffer ready: {len(offline_buffer)} transitions.")

    cfg = RLPDConfig(
        env_name="stk",
        seed=args.seed,
        max_steps=args.max_steps,
        n_envs=1,
        steps_per_round=1,
        updates_per_round=args.utd,
        n_critics=args.n_critics,
        critic_subset_size=args.critic_subset,
        entropy_mode="fixed" if args.alpha is not None else "auto",
        init_entropy_coef=args.alpha if args.alpha is not None else args.init_alpha,
        max_grad_norm=args.max_grad_norm,
        reward_scale=args.reward_scale,
        learning_starts=args.learning_starts,
    )
    print(f"UTD ratio = {cfg.updates_per_round / cfg.steps_per_round:g}, "
          f"E={cfg.n_critics}, Z={cfg.critic_subset_size}, tau={cfg.tau}, "
          f"alpha={cfg.entropy_mode}:{cfg.init_entropy_coef}, clip={cfg.max_grad_norm}, "
          f"reward_scale={cfg.reward_scale}, learning_starts={cfg.learning_starts}")

    run_dir = args.run_dir or f"runs/rlpd-{args.expert}-S{args.seed}"
    evaluator = run_rlpd(cfg, offline_buffer=offline_buffer, stk_env=env, run_dir=run_dir,
                         make_evaluator=_evaluator_factory(env, args))
    _save_actor(evaluator.best_actor, run_dir, also_submission=not args.no_submit_weights)


def cmd_bc(args: argparse.Namespace) -> None:
    """Behaviour cloning: regress actor.act(obs) onto the clean expert action.

    Uses the same cached dataset as RLPD (``expert_action`` labels rather than
    the noisy executed actions) and the same evaluation suite, so the curves
    in runs/bc-* and runs/rlpd-* are directly comparable.
    """
    import torch
    import torch.nn.functional as F
    from torch.utils.tensorboard import SummaryWriter
    from tqdm.auto import tqdm
    from .sac import SACConfig, SquashedGaussianActor
    from .stk_bridge import STKVecEnv, OBS_DIM

    torch.manual_seed(args.seed)
    env = STKVecEnv(track=args.track, num_kart=args.num_kart, seed=args.seed)
    dataset = _get_dataset(env, args)
    obs, target = dataset["obs"], dataset["expert_action"]
    n = len(obs)

    hidden = SACConfig.actor_hidden
    actor = SquashedGaussianActor(OBS_DIM, hidden, target.shape[1])
    optimizer = torch.optim.Adam(actor.parameters(), lr=args.lr)

    run_dir = args.run_dir or f"runs/bc-{args.expert}-S{args.seed}"
    writer = SummaryWriter(run_dir)
    evaluator = _evaluator_factory(env, args)(writer)

    for step in tqdm(range(1, args.grad_steps + 1)):
        idx = torch.randint(n, (args.batch_size,))
        loss = F.mse_loss(actor.act(obs[idx]), target[idx])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % 100 == 0:
            writer.add_scalar("loss/bc", loss.item(), step)
        if result := evaluator.run_if_needed(step, actor):
            tqdm.write(f"step {step}: eval={result.mean:.1f} best={evaluator.best_reward:.1f}")

    writer.close()
    _save_actor(evaluator.best_actor, run_dir, also_submission=False)


def cmd_dagger(args: argparse.Namespace) -> None:
    """DAgger on the same cached expert dataset and evaluation suite as RLPD."""
    from torch.utils.tensorboard import SummaryWriter
    from .dagger import DAggerConfig, run_dagger
    from .stk_bridge import STKVecEnv

    env = STKVecEnv(track=args.track, num_kart=args.num_kart, seed=args.seed)
    dataset = _get_dataset(env, args)

    cfg = DAggerConfig(
        seed=args.seed,
        max_steps=args.max_steps,
        steps_per_iter=args.steps_per_iter,
        beta0=args.beta0,
        beta_decay=args.beta_decay,
        initial_grad_steps=args.initial_grad_steps,
        grad_steps_per_iter=args.grad_steps_per_iter,
    )
    print(cfg)

    run_dir = args.run_dir or f"runs/dagger-{args.expert}-S{args.seed}"
    writer = SummaryWriter(run_dir)
    evaluator = run_dagger(cfg, env.flat_env, dataset, _resolve_expert(args.expert),
                           writer, _evaluator_factory(env, args)(writer))
    writer.close()
    _save_actor(evaluator.best_actor, run_dir, also_submission=not args.no_submit_weights)


def cmd_compare(args: argparse.Namespace) -> None:
    """Evaluate the expert and each run's actor.pth on the same track suite."""
    import torch
    from .sac import SACConfig, SquashedGaussianActor
    from .stk_bridge import make_stk_flat_env, OBS_DIM, ACTION_DIM
    from .stk_eval import run_suite, actor_factory, expert_factory, format_table

    env = make_stk_flat_env(num_kart=args.num_kart)
    results = {}
    if args.expert:
        print(f"Evaluating expert {args.expert}…")
        results[f"expert-{args.expert}"] = run_suite(
            env, expert_factory(_resolve_expert(args.expert)), args.eval_tracks, args.eval_seed)
    for run in args.runs:
        path = Path(run)
        if path.is_dir():
            path = path / "actor.pth"
        actor = SquashedGaussianActor(OBS_DIM, SACConfig.actor_hidden, ACTION_DIM)
        actor.load_state_dict(torch.load(path, map_location="cpu"))
        actor.eval()
        print(f"Evaluating {path}…")
        results[path.parent.name] = run_suite(
            env, actor_factory(actor), args.eval_tracks, args.eval_seed)

    print()
    print(format_table(results))
    print("\n(* = lap finished; one episode per track, seed "
          f"{args.eval_seed}; num_kart={args.num_kart})")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_dataset(env, args) -> dict:
    """Load the cached expert dataset, or collect and cache it."""
    from .stk_bridge import (
        collect_expert_dataset, save_dataset, load_dataset, OBS_DIM, ACTION_DIM,
    )

    name = f"{args.expert}-{args.offline_steps}-obs{OBS_DIM}-act{ACTION_DIM}"
    if args.track:
        import hashlib
        tracks = sorted(args.track)
        tag = tracks[0] if len(tracks) == 1 else \
            f"{len(tracks)}tracks-{hashlib.md5(','.join(tracks).encode()).hexdigest()[:8]}"
        name += f"-{tag}"
    path = DATA_DIR / f"{name}.pt"
    if path.exists():
        print(f"Loading expert dataset {path}")
        dataset = load_dataset(path)
    else:
        print(f"Collecting {args.offline_steps} expert steps from {args.expert}…")
        dataset = collect_expert_dataset(
            env.flat_env, _resolve_expert(args.expert), n_steps=args.offline_steps
        )
        dataset["meta"]["tracks"] = args.track  # None = all tracks
        save_dataset(dataset, path)
        print(f"Saved expert dataset → {path}")
    assert dataset["obs"].shape[1] == OBS_DIM, "dataset built with another observation layout"
    assert dataset["action"].shape[1] == ACTION_DIM, "dataset built with another action layout"
    eps = dataset["meta"]["episodes"]
    if eps:
        print(f"  {len(eps)} expert episodes, "
              f"{sum(e['finished'] for e in eps)} finished, "
              f"mean return {sum(e['return'] for e in eps) / len(eps):.1f}")
    return dataset


def _resolve_expert(name: str):
    """Return the KartAgent subclass for the given team name."""
    import sys, os, importlib

    # 1. Lock in the installed pystk2_gymnasium BEFORE adding expert/ to the
    #    path, so the local expert/pystk2_gymnasium/ never shadows the venv copy.
    import pystk2_gymnasium  # noqa: F401

    # 2. Now safe to add expert/ — agents and utils live there, but
    #    pystk2_gymnasium is already cached so Python won't re-import it.
    expert_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "expert"))
    if expert_dir not in sys.path:
        sys.path.insert(0, expert_dir)

    table = {
        "team1": ("agents.team1.agent1", "Agent1"),
        "team2": ("agents.team2.agent2", "Agent2"),
        "team3": ("agents.team3.agent3", "Agent3"),
        "team4": ("agents.team4.agent4", "Agent4"),
        "team5": ("agents.team5.agent5", "Agent5"),
    }
    if name not in table:
        raise ValueError(f"Unknown expert '{name}'. Choose from: {list(table)}")
    module_name, cls_name = table[name]
    module = importlib.import_module(module_name)
    return getattr(module, cls_name)


def _save_actor(actor, run_dir: str, also_submission: bool = True) -> None:
    """Save the best actor to <run_dir>/actor.pth (and to the submission
    weights stk_actor/pystk_actor.pth for RL runs)."""
    import torch
    if actor is None:
        print("No best actor to save (training may not have evaluated yet).")
        return
    Path(run_dir).mkdir(parents=True, exist_ok=True)
    outs = [Path(run_dir) / "actor.pth"]
    if also_submission:
        outs.append(Path(__file__).parent / "pystk_actor.pth")
    for out in outs:
        torch.save(actor.state_dict(), out)
        print(f"Saved best actor → {out}")


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m stk_actor.learn")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_sac = sub.add_parser("sac", help="Train SAC from scratch")
    _add_common_args(p_sac)

    p_rlpd = sub.add_parser("rlpd", help="Train RLPD with expert offline data")
    _add_common_args(p_rlpd)
    _add_expert_args(p_rlpd)
    p_rlpd.add_argument("--utd", type=int, default=10,
                        help="Gradient updates per env step (paper: 20)")
    p_rlpd.add_argument("--n-critics", type=int, default=10, help="Ensemble size E")
    p_rlpd.add_argument("--critic-subset", type=int, default=2,
                        help="Target subset size Z")
    p_rlpd.add_argument("--alpha", type=float, default=None,
                        help="Fixed entropy coefficient (default: auto-tuned)")
    p_rlpd.add_argument("--init-alpha", type=float, default=0.2,
                        help="Initial entropy coefficient when auto-tuned")
    p_rlpd.add_argument("--max-grad-norm", type=float, default=0.5)
    p_rlpd.add_argument("--reward-scale", type=float, default=1.0,
                        help="Multiply rewards in the critic target")
    p_rlpd.add_argument("--learning-starts", type=int, default=256,
                        help="Online transitions collected before the first update")

    p_bc = sub.add_parser("bc", help="Behaviour cloning on the expert data")
    _add_common_args(p_bc)
    _add_expert_args(p_bc)
    p_bc.add_argument("--grad-steps", type=int, default=50_000)
    p_bc.add_argument("--batch-size", type=int, default=256)
    p_bc.add_argument("--lr", type=float, default=3e-4)
    p_bc.set_defaults(eval_every=5_000)

    p_dagger = sub.add_parser("dagger", help="Train DAgger with the expert")
    _add_common_args(p_dagger)
    _add_expert_args(p_dagger)
    p_dagger.add_argument("--steps-per-iter", type=int, default=2_000,
                          help="Env steps collected per DAgger iteration")
    p_dagger.add_argument("--beta0", type=float, default=0.5,
                          help="Probability of executing the expert action at iteration 1")
    p_dagger.add_argument("--beta-decay", type=float, default=0.5,
                          help="beta_i = beta0 * beta_decay ** (i - 1)")
    p_dagger.add_argument("--initial-grad-steps", type=int, default=5_000,
                          help="Behaviour cloning steps on the expert data before iteration 1")
    p_dagger.add_argument("--grad-steps-per-iter", type=int, default=1_000)

    p_cmp = sub.add_parser("compare", help="Evaluate the expert and saved actors")
    p_cmp.add_argument("--runs", nargs="+", default=[],
                       help="Run directories (or .pth files) to evaluate")
    p_cmp.add_argument("--expert", default="team3",
                       help="Expert to include as reference ('' to skip)")
    p_cmp.add_argument("--num-kart", type=int, default=1)
    _add_eval_args(p_cmp)

    args = parser.parse_args()
    {"sac": cmd_sac, "rlpd": cmd_rlpd, "bc": cmd_bc, "dagger": cmd_dagger,
     "compare": cmd_compare}[args.cmd](args)


if __name__ == "__main__":
    main()
