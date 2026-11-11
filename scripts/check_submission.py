#!/usr/bin/env -S uv run --script
# /// script
# requires-python = "==3.12.*"
# dependencies = [
#     "pystk2-gymnasium==0.10.0",
#     "gymnasium==1.3.0",
#     "numpy==2.5.3",
#     "torch==2.12.0",
#     "stable-baselines3==2.9.0",
#     "sb3-contrib==2.9.0",
# ]
# ///
"""Checks that the stk_actor/ folder can be submitted on ML-Arena.

Run it with:

    uv run scripts/check_submission.py

It runs with the libraries of the platform runtime (PyTorch · SuperTuxKart):
it checks the upload limits, copies the files to upload to a temporary
folder (flat, as the platform does), then loads the agent through the platform shim (agent.py) and plays a short race,
reporting the time per step.
"""

import argparse
import importlib.util
import resource
import statistics
import sys
import shutil
import tempfile
import time
from pathlib import Path

from submission_files import AGENT_DIR, CheckError, collect_files

#: Platform limits (see https://ml-arena.com/viewchallenge/169)
PYSTK2_GYMNASIUM_VERSION = "0.10.0"
MAX_TIME_PER_STEP = 1.0
MAX_MEMORY = 3 * 1024**3
NUM_KARTS = 4


def copy_files(folder: Path, upload_dir: Path):
    """Copies the files to upload, checking the platform limits"""
    files = collect_files(folder)
    print(f"Files to upload from {folder}/:")
    for path in files:
        print(f"  {path.name:30s} {path.stat().st_size / 1024**2:8.2f} MB")
        shutil.copy(path, upload_dir / path.name)


def load_agent(upload_dir: Path):
    """Loads the shim from the upload folder, as the platform does"""
    spec = importlib.util.spec_from_file_location(
        "ml_arena_agent", upload_dir / "agent.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def race(module, steps: int, track: str | None):
    """Plays a short race with the agent, returns the time of each step"""
    import gymnasium as gym
    from pystk2_gymnasium import AgentSpec

    agent = module.Agent()
    info = agent.setup(
        {
            "protocol": module.PROTOCOL,
            "env_id": module.ENV_ID,
            "pystk2_gymnasium_version": PYSTK2_GYMNASIUM_VERSION,
        }
    )
    print(f"Agent loaded: {info}")

    env = gym.make(
        module.ENV_ID,
        agents=[AgentSpec(name=info["player_name"])],
        num_kart=NUM_KARTS,
        track=track,
    )
    try:
        obs, _ = env.reset()
        print(f"Racing on {env.unwrapped.current_track} for {steps} steps")
        durations = []
        for _ in range(steps):
            start = time.perf_counter()
            action = agent.act(obs["0"])
            durations.append(time.perf_counter() - start)
            obs, _, terminated, truncated, _ = env.step({"0": action})
            if terminated or truncated:
                break
    finally:
        env.close()
    return durations


def check(folder: Path, steps: int, track: str | None):
    with tempfile.TemporaryDirectory() as tmp:
        upload_dir = Path(tmp)
        copy_files(folder, upload_dir)
        durations = race(load_agent(upload_dir), steps, track)

    mean, worst = statistics.mean(durations), max(durations)
    print(f"Time per step: mean {mean * 1000:.1f} ms, max {worst * 1000:.1f} ms")
    # ru_maxrss is in bytes on macOS, in kilobytes on Linux
    memory = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    memory *= 1 if sys.platform == "darwin" else 1024
    print(f"Peak memory: {memory / 1024**2:.0f} MB")

    if worst > MAX_TIME_PER_STEP:
        raise CheckError(
            f"a step took {worst:.2f} s, the platform allows {MAX_TIME_PER_STEP} s"
            " (on CPU only)"
        )
    if memory > MAX_MEMORY:
        raise CheckError(
            f"the agent used {memory / 1024**3:.1f} GB, the platform allows "
            f"{MAX_MEMORY / 1024**3:.0f} GB"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "folder", nargs="?", type=Path, default=AGENT_DIR, help="Agent folder"
    )
    parser.add_argument("--steps", type=int, default=100, help="Number of steps")
    parser.add_argument("--track", help="Track (random by default)")
    args = parser.parse_args()

    try:
        check(args.folder, args.steps, args.track)
    except CheckError as e:
        print(f"ERROR: {e}")
        sys.exit(1)
    print("OK: the submission looks valid")


if __name__ == "__main__":
    main()
