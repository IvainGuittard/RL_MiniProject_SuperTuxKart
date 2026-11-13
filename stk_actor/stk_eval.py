"""Evaluation on a fixed suite of tracks (same tracks, same seed, every time).

pystk2 is a per-process singleton, so evaluation runs on the *training* env
(``STKVecEnv.flat_env``): the track is forced through
``env.unwrapped.default_track`` for the duration of the evaluation, then put
back. Since evaluation resets that env, the training loop must restart its
own episode afterwards (see ``collector._obs = None`` in rlpd.py / sac.py).

Every evaluation runs one deterministic episode per track with
``reset(seed=seed)``, so two evaluations differ only by the policy.
"""

from __future__ import annotations

import copy
from pathlib import Path
from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from rl_mind.evaluation import EvalResult

from .stk_bridge import STKFlatWrapper, expert_to_env


DEFAULT_EVAL_TRACKS = (
    "zengarden",            #  710 m, short and technical
    "lighthouse",           #  880 m
    "hacienda",             # 1167 m
    "cornfield_crossing",   # 1264 m
    "scotland",             # 1329 m
)

# A policy factory is called after each reset (an expert needs the fresh env)
# and returns ``flat_obs -> action`` (ACTION_DIM array, or a full STK dict
# for an expert).
PolicyFactory = Callable[[STKFlatWrapper], Callable[[np.ndarray], np.ndarray]]


@dataclass(frozen=True)
class EpisodeStats:
    track: str
    ret: float
    finished: bool
    steps: int


def actor_factory(actor) -> PolicyFactory:
    """Deterministic behaviour (``actor.act``) of a SquashedGaussianActor."""
    def factory(env):
        def policy(obs):
            with torch.no_grad():
                obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
                return actor.act(obs_t)[0].numpy()
        return policy
    return factory


def expert_factory(expert_cls) -> PolicyFactory:
    """Rule-based expert, executing its own full action."""
    def factory(env):
        expert = expert_cls(env)
        return lambda obs: expert_to_env(expert.choose_action(env.raw_obs))
    return factory


def run_suite(
    env: STKFlatWrapper,
    factory: PolicyFactory,
    tracks=DEFAULT_EVAL_TRACKS,
    seed: int = 0,
) -> list[EpisodeStats]:
    """One deterministic episode per track; restores the env's track setting."""
    unwrapped = env.unwrapped
    saved_track = unwrapped.default_track
    saved_pool = getattr(env, "track_pool", None)
    env.track_pool = None  # the evaluation chooses its own tracks
    stats = []
    try:
        for track in tracks:
            unwrapped.default_track = track
            obs, _ = env.reset(seed=seed)
            policy = factory(env)
            ret, steps, done, terminated = 0.0, 0, False, False
            while not done:
                obs, reward, terminated, truncated, _ = env.step(policy(obs))
                ret += float(reward)
                steps += 1
                done = terminated or truncated
            stats.append(EpisodeStats(track, ret, bool(terminated), steps))
    finally:
        unwrapped.default_track = saved_track
        env.track_pool = saved_pool
    return stats


class TrackSuiteEvaluator:
    """Drop-in replacement for rl_mind's Evaluator, on a fixed track suite.

    Keeps ``history``, ``best_actor`` and ``best_reward`` like the original,
    and logs to TensorBoard:

    - ``eval/steps``: mean episode length over the suite (what best_actor is
      chosen on: fewer is better)
    - ``eval/reward``: mean return over the suite
    - ``eval/finished``: fraction of tracks where the lap was completed
    - ``eval_track/<track>`` and ``eval_steps/<track>``: return and steps on
      each track
    """

    def __init__(
        self,
        env: STKFlatWrapper,
        every: int,
        tracks=DEFAULT_EVAL_TRACKS,
        seed: int = 0,
        writer: SummaryWriter | None = None,
        save_dir=None,
    ):
        self.env = env
        self.every = every
        self.tracks = tuple(tracks)
        self.seed = seed
        self.writer = writer
        # If set, <save_dir>/actor.pth (best so far) and actor_last.pth are
        # written at every evaluation, so a killed run keeps its weights.
        self.save_dir = Path(save_dir) if save_dir is not None else None
        self.history: list[EvalResult] = []
        self.best_actor = None
        self.best_reward = -float("inf")  # mean return of best_actor (for display)
        self.best_steps = float("inf")    # selection criterion, see run_if_needed
        self._last_eval_step = 0

    def run(self, actor) -> list[EpisodeStats]:
        return run_suite(self.env, actor_factory(actor), self.tracks, self.seed)

    def run_if_needed(self, steps: int, actor) -> EvalResult | None:
        if steps - self._last_eval_step < self.every:
            return None
        self._last_eval_step = steps

        stats = self.run(actor)
        rewards = torch.tensor([s.ret for s in stats])
        mean = float(rewards.mean())
        # The best actor is the one with the fewest steps per lap on average.
        mean_steps = float(np.mean([s.steps for s in stats]))
        is_best = mean_steps < self.best_steps
        if is_best:
            self.best_steps = mean_steps
            self.best_reward = mean
            self.best_actor = copy.deepcopy(actor)

        result = EvalResult(step=steps, rewards=rewards, is_best=is_best)
        self.history.append(result)
        if self.save_dir is not None:
            self.save_dir.mkdir(parents=True, exist_ok=True)
            torch.save(actor.state_dict(), self.save_dir / "actor_last.pth")
            if is_best:
                torch.save(actor.state_dict(), self.save_dir / "actor.pth")
        if self.writer is not None:
            self.writer.add_scalar("eval/reward", mean, steps)
            self.writer.add_scalar("eval/steps", mean_steps, steps)
            self.writer.add_scalar(
                "eval/finished", float(np.mean([s.finished for s in stats])), steps
            )
            for s in stats:
                self.writer.add_scalar(f"eval_track/{s.track}", s.ret, steps)
                self.writer.add_scalar(f"eval_steps/{s.track}", s.steps, steps)
        return result


def format_table(results: dict[str, list[EpisodeStats]]) -> str:
    """Rows = policies, columns = tracks (return, * = lap finished)."""
    names = list(results)
    tracks = [s.track for s in results[names[0]]]
    w = max(len(n) for n in names) + 2
    header = "policy".ljust(w) + "".join(t[:12].rjust(13) for t in tracks)
    header += "mean".rjust(9) + "finished".rjust(10)
    lines = [header, "-" * len(header)]
    for name in names:
        stats = results[name]
        row = name.ljust(w)
        row += "".join(f"{s.ret:8.1f}{'*' if s.finished else ' '}".rjust(13) for s in stats)
        row += f"{np.mean([s.ret for s in stats]):9.1f}"
        row += f"{sum(s.finished for s in stats):>6d}/{len(stats)}"
        lines.append(row)
    return "\n".join(lines)
