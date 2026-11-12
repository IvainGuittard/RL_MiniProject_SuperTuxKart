"""Bridge between pystk2-gymnasium and the rl_mind SAC / RLPD pipeline.

Three pieces:
  1. STKFlatWrapper   — gymnasium wrapper: fully flat float obs + 7-dim
                        continuous action space (covering all STK actions).
  2. STKVecEnv        — rl_mind-compatible vectorised env (single instance;
                        pystk2 is a per-process singleton).
  3. collect_expert_data — run any KartAgent subclass, fill a ReplayBuffer.

Observation pipeline (applied before flattening):
  supertuxkart/full-v0
    └─ ConstantSizedObservations  (pad variable-length sequences)
        └─ PolarObservations       (3-D positions → polar coords)
            └─ STKFlatWrapper      (Dict obs → flat float32 Box)

This is exactly the "supertuxkart/simple-v0" stack plus a final flatten step.

Action encoding (7 dims, all in [-1, 1]):
  dim 0  acceleration  mapped to [0, 1] by (x+1)/2
  dim 1  steer         kept in [-1, 1]
  dim 2  brake         x > 0 → 1, else 0
  dim 3  drift         x > 0 → 1, else 0
  dim 4  fire          x > 0 → 1, else 0
  dim 5  nitro         x > 0 → 1, else 0
  dim 6  rescue        x > 0 → 1, else 0
"""

from __future__ import annotations

import numpy as np
import torch
import gymnasium as gym
from gymnasium import spaces
from torch import Tensor

from rl_mind.env import EnvStep
from rl_mind.data import ReplayBuffer, Transitions
from rl_mind.core import Action


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_ACTION_KEYS = ["acceleration", "steer", "brake", "drift", "fire", "nitro", "rescue"]


def _space_flat_dim(space: gym.Space) -> int:
    """Number of float32 scalars a space flattens to."""
    if isinstance(space, spaces.Text):
        return 0
    if isinstance(space, (spaces.Discrete,)):
        return 1
    if isinstance(space, spaces.MultiDiscrete):
        return int(len(space.nvec))
    if isinstance(space, spaces.Box):
        return int(np.prod(space.shape)) if space.shape else 1
    return 0


def _obs_to_flat(obs: dict, keys: list[str]) -> np.ndarray:
    """Flatten a Dict observation to a float32 numpy vector."""
    parts = []
    for key in keys:
        parts.append(np.asarray(obs[key], dtype=np.float32).flatten())
    return np.concatenate(parts)


def expert_action_to_flat(action: dict) -> np.ndarray:
    """Convert an STK dict action to a 7-dim float32 array in [-1, 1].

    Inverse of the action decoding in STKFlatWrapper: used when storing
    expert transitions so they live in the same space as SAC actions.
    """
    accel = float(np.asarray(action["acceleration"]).flat[0])
    steer = float(np.asarray(action["steer"]).flat[0])
    return np.array([
        2.0 * accel - 1.0,
        steer,
        1.0 if action.get("brake", False) else -1.0,
        1.0 if action.get("drift", False) else -1.0,
        1.0 if action.get("fire",  False) else -1.0,
        1.0 if action.get("nitro", False) else -1.0,
        1.0 if action.get("rescue",False) else -1.0,
    ], dtype=np.float32)


# ---------------------------------------------------------------------------
# 1. STKFlatWrapper
# ---------------------------------------------------------------------------

class STKFlatWrapper(gym.Wrapper):
    """Fully flat obs (float32 Box) + 7-dim continuous action space.

    Expects an env that already has ConstantSizedObservations and
    PolarObservations applied (i.e. supertuxkart/simple-v0 or similar).
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)

        dict_space = env.observation_space
        assert isinstance(dict_space, spaces.Dict), (
            "STKFlatWrapper expects a Dict observation space; "
            "wrap supertuxkart/simple-v0, not full-v0."
        )

        # Sorted keys, skipping non-numeric (Text) fields
        self._obs_keys = sorted(
            k for k, sp in dict_space.spaces.items()
            if _space_flat_dim(sp) > 0
        )
        obs_dim = sum(_space_flat_dim(dict_space[k]) for k in self._obs_keys)

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(7,), dtype=np.float32
        )

    # -- gymnasium Wrapper protocol --

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return _obs_to_flat(obs, self._obs_keys), info

    def step(self, action):
        stk_action = self._decode_action(action)
        obs, reward, terminated, truncated, info = self.env.step(stk_action)
        return _obs_to_flat(obs, self._obs_keys), reward, terminated, truncated, info

    # -- helpers --

    @staticmethod
    def _decode_action(action: np.ndarray) -> dict:
        """Map a 7-dim float array in [-1,1] to an STK action dict."""
        return {
            "acceleration": np.array([(float(action[0]) + 1.0) / 2.0]),
            "steer":        np.array([float(action[1])]),
            "brake":        int(action[2] > 0),
            "drift":        int(action[3] > 0),
            "fire":         int(action[4] > 0),
            "nitro":        int(action[5] > 0),
            "rescue":       int(action[6] > 0),
        }


def make_stk_flat_env(track: str | None = None, num_kart: int = 3) -> gym.Env:
    """Create a fully-flat STK env ready for SAC/RLPD training."""
    import pystk2_gymnasium  # noqa: F401 — registers the envs
    env = gym.make(
        "supertuxkart/simple-v0",
        track=track,
        num_kart=num_kart,
        render_mode=None,
    )
    return STKFlatWrapper(env)


# ---------------------------------------------------------------------------
# 2. STKVecEnv — rl_mind VecEnv interface, single STK instance
# ---------------------------------------------------------------------------

class STKVecEnv:
    """Single-instance STK environment with the rl_mind VecEnv interface.

    pystk2 is a per-process singleton, so num_envs is always 1.
    Uses same-step-reset semantics: when an episode ends, the terminal obs
    is returned in next_obs and the first obs of the new episode is in obs,
    all within the same call to step(). No "ghost" auto-reset step is emitted.
    """

    #: pystk2 supports only one instance per process
    num_envs: int = 1
    #: Terminal and first obs of next episode are both in the same step
    same_step_reset: bool = True

    def __init__(
        self,
        track: str | None = None,
        num_kart: int = 3,
        seed: int | None = None,
    ):
        self._env = make_stk_flat_env(track=track, num_kart=num_kart)
        self._seed = seed

        self.observation_space: spaces.Box = self._env.observation_space
        self.action_space: spaces.Box = self._env.action_space

    @property
    def observation_dim(self) -> int:
        return int(np.prod(self.observation_space.shape))

    @property
    def action_dim(self) -> int:
        return int(np.prod(self.action_space.shape))

    @property
    def is_continuous(self) -> bool:
        return True

    def reset(self) -> Tensor:
        """Reset the environment; return initial obs as `[1, obs_dim]` tensor."""
        obs, _ = self._env.reset(seed=self._seed)
        self._seed = None
        return torch.from_numpy(obs).unsqueeze(0)

    def step(self, actions: Tensor) -> EnvStep:
        """Perform one step; return an EnvStep with batch dimension 1.

        Same-step reset: when done, obs holds the first obs of the new episode
        and next_obs holds the terminal obs of the ended one.
        """
        action_np = actions[0].detach().cpu().numpy()
        next_obs_np, reward, terminated, truncated, _ = self._env.step(action_np)

        next_obs_t = torch.from_numpy(next_obs_np).unsqueeze(0)

        if terminated or truncated:
            obs_np, _ = self._env.reset()
            obs_t = torch.from_numpy(obs_np).unsqueeze(0)
        else:
            obs_t = next_obs_t

        return EnvStep(
            obs=obs_t,
            next_obs=next_obs_t,
            reward=torch.tensor([reward], dtype=torch.float32),
            terminated=torch.tensor([terminated], dtype=torch.bool),
            truncated=torch.tensor([truncated], dtype=torch.bool),
        )

    def close(self) -> None:
        self._env.close()


# ---------------------------------------------------------------------------
# 3. collect_expert_data
# ---------------------------------------------------------------------------

class _STKObsProcessor:
    """Stateless pipeline: raw full-v0 obs dict → flat float32 numpy array.

    Replicates ConstantSizedObservations + PolarObservations + flatten,
    matching the observation produced by STKFlatWrapper (same key order,
    same padding sizes).
    """

    def __init__(
        self,
        raw_env: gym.Env,
        state_items: int = 5,
        state_karts: int = 5,
        state_paths: int = 5,
    ):
        from pystk2_gymnasium.stk_wrappers import (
            ConstantSizedObservations,
            PolarObservations,
        )
        csized = ConstantSizedObservations(
            raw_env,
            state_items=state_items,
            state_karts=state_karts,
            state_paths=state_paths,
        )
        polar = PolarObservations(csized)

        self._csized_obs = csized.observation
        self._polar_obs = polar.observation
        self._keys = sorted(
            k for k, sp in polar.observation_space.spaces.items()
            if _space_flat_dim(sp) > 0
        )
        self.obs_dim = sum(
            _space_flat_dim(polar.observation_space[k]) for k in self._keys
        )

    def __call__(self, raw_obs: dict) -> np.ndarray:
        obs = self._csized_obs(raw_obs)
        obs = self._polar_obs(obs)
        return _obs_to_flat(obs, self._keys)


def collect_expert_data(
    agent_cls,
    n_steps: int,
    track: str | None = None,
    num_kart: int = 3,
    **agent_kwargs,
) -> ReplayBuffer:
    """Run a KartAgent expert for n_steps, return a populated ReplayBuffer.

    The returned buffer stores flat observations and 7-dim continuous actions
    in the same space as the SAC / RLPD training envs, so it can be directly
    passed to run_rlpd() as the offline_buffer.

    :param agent_cls: A KartAgent subclass (e.g. Agent1).
    :param n_steps:   Number of transitions to collect.
    :param track:     Track name (None = random per episode).
    :param num_kart:  Number of karts in the race.
    :param agent_kwargs: Extra keyword arguments forwarded to agent_cls().
    :returns: A ReplayBuffer of size n_steps.
    """
    # pystk2_gymnasium must already be cached from the venv at this point
    # (the caller imports it before reaching here via learn.py).
    import pystk2_gymnasium  # noqa: F401 — registers envs if not yet done

    raw_env = gym.make(
        "supertuxkart/full-v0",
        track=track,
        num_kart=num_kart,
        render_mode=None,
    )

    processor = _STKObsProcessor(raw_env.unwrapped)
    expert = agent_cls(raw_env, **agent_kwargs)

    buffer: ReplayBuffer = ReplayBuffer(n_steps)

    obs_raw, _ = raw_env.reset()
    # Sync the expert's internal state
    expert.obs = obs_raw
    expert.agent_positions = []

    collected = 0
    while collected < n_steps:
        flat_obs = processor(obs_raw)
        action_dict = expert.choose_action(obs_raw)
        flat_action = expert_action_to_flat(action_dict)

        next_obs_raw, reward, terminated, truncated, _ = raw_env.step(action_dict)
        flat_next_obs = processor(next_obs_raw)

        transitions = Transitions(
            obs=torch.from_numpy(flat_obs).unsqueeze(0),
            action=Action(value=torch.from_numpy(flat_action).unsqueeze(0)),
            reward=torch.tensor([float(reward)], dtype=torch.float32),
            next_obs=torch.from_numpy(flat_next_obs).unsqueeze(0),
            terminated=torch.tensor([terminated], dtype=torch.bool),
        )
        buffer.add(transitions)
        collected += 1

        obs_raw = next_obs_raw
        if terminated or truncated:
            obs_raw, _ = raw_env.reset()
            expert.obs = obs_raw
            expert.agent_positions = []

    raw_env.close()
    return buffer
