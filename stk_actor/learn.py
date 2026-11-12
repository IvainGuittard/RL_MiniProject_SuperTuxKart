"""Training entry point.

Modes
-----
sac    — train SAC from scratch on the STK flat env (baseline)
rlpd   — collect expert data, then train RLPD (SAC + offline expert buffer)

Run from the repository root:

    PYTHONPATH=expert python -m stk_actor.learn sac
    PYTHONPATH=expert python -m stk_actor.learn rlpd --expert team1 --offline-steps 50000
"""

import argparse
import sys
from pathlib import Path


def _add_common_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--track", default=None, help="Track name (default: random)")
    p.add_argument("--num-kart", type=int, default=3)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--max-steps", type=int, default=300_000)
    p.add_argument("--run-dir", default=None, help="TensorBoard / checkpoint directory")


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
    evaluator = run_sac(cfg, stk_env=env, run_dir=run_dir)

    _save_actor(evaluator, run_dir)


def cmd_rlpd(args: argparse.Namespace) -> None:
    from .rlpd import RLPDConfig, run_rlpd
    from .stk_bridge import STKVecEnv, collect_expert_data

    expert_cls = _resolve_expert(args.expert)

    print(f"Collecting {args.offline_steps} expert steps from {args.expert}…")
    offline_buffer = collect_expert_data(
        expert_cls,
        n_steps=args.offline_steps,
        track=args.track,
        num_kart=args.num_kart,
    )
    print(f"Offline buffer ready: {len(offline_buffer)} transitions.")

    env = STKVecEnv(track=args.track, num_kart=args.num_kart, seed=args.seed)

    cfg = RLPDConfig(
        env_name="stk",
        seed=args.seed,
        max_steps=args.max_steps,
        n_envs=1,
        steps_per_round=1,
        updates_per_round=1,
    )

    run_dir = args.run_dir or f"runs/rlpd-{args.expert}-S{args.seed}"
    evaluator = run_rlpd(cfg, offline_buffer=offline_buffer, stk_env=env, run_dir=run_dir)

    _save_actor(evaluator, run_dir)


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
    import importlib
    module = importlib.import_module(module_name)
    return getattr(module, cls_name)


def _save_actor(evaluator, run_dir: str) -> None:
    import torch
    mod_path = Path(__file__).parent
    out = mod_path / "pystk_actor.pth"
    if hasattr(evaluator, "best_actor") and evaluator.best_actor is not None:
        torch.save(evaluator.best_actor.state_dict(), out)
        print(f"Saved best actor → {out}")
    else:
        print("No best actor to save (training may not have evaluated yet).")


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m stk_actor.learn")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_sac = sub.add_parser("sac", help="Train SAC from scratch")
    _add_common_args(p_sac)

    p_rlpd = sub.add_parser("rlpd", help="Train RLPD with expert offline data")
    _add_common_args(p_rlpd)
    p_rlpd.add_argument(
        "--expert", default="team1",
        choices=["team1", "team2", "team3", "team4", "team5"],
        help="Which expert agent to use as the offline data source",
    )
    p_rlpd.add_argument(
        "--offline-steps", type=int, default=50_000,
        help="Number of expert transitions to collect",
    )

    args = parser.parse_args()
    {"sac": cmd_sac, "rlpd": cmd_rlpd}[args.cmd](args)


if __name__ == "__main__":
    main()
