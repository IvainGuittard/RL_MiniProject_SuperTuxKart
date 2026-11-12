"""Bridge between pystk2-gymnasium and the rl_mind SAC / RLPD / DAgger pipeline.

Uses supertuxkart/full-v0 directly — no ConstantSizedObservations or
PolarObservations dependency.  Variable-length path sequences are padded here,
so num_kart=1 works without hitting the installed-wrapper np.stack crash.

Observation (flat float32, OBS_DIM = 11 + N_PATHS * 7):
  velocity              3   kart velocity in world frame
  distance_down_track   1   progress along track
  center_path_distance  1   lateral offset from track centre
  front                 3   kart heading direction
  center_path           3   nearest track-centre position
  paths_start      N×3     upcoming track-node start positions (padded)
  paths_end        N×3     upcoming track-node end positions
  paths_width      N×1     upcoming track-node widths

Action (5 dims, all in [-1, 1]):
  0  acceleration   mapped to [0, 1] by (x+1)/2
  1  steer          kept in [-1, 1]
  2  brake          x > 0 → 1, else 0
  3  drift          x > 0 → 1, else 0
  4  nitro          x > 0 → 1, else 0
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


N_PATHS = 5                     # lookahead track nodes
OBS_DIM = 11 + N_PATHS * 7     # 46 with N_PATHS=5


# ---------------------------------------------------------------------------
# Observation / action helpers
# ---------------------------------------------------------------------------

def _pad_sequence(seq, n: int, dim: int) -> np.ndarray:
    """Pad a variable-length sequence of dim-vectors to exactly n rows."""
    out = np.zeros((n, dim), dtype=np.float32)
    for i, x in enumerate(seq[:n]):
        out[i] = np.asarray(x, dtype=np.float32)
    return out.flatten()


def obs_to_flat(obs: dict) -> np.ndarray:
    """Convert a full-v0 Dict observation to a flat float32 vector."""
    return np.concatenate([
        np.asarray(obs["velocity"],             dtype=np.float32).flatten(),  # 3
        np.asarray(obs["distance_down_track"],  dtype=np.float32).flatten(),  # 1
        np.asarray(obs["center_path_distance"], dtype=np.float32).flatten(),  # 1
        np.asarray(obs["front"],                dtype=np.float32).flatten(),  # 3
        np.asarray(obs["center_path"],          dtype=np.float32).flatten(),  # 3
        _pad_sequence(obs["paths_start"], N_PATHS, 3),                        # N*3
        _pad_sequence(obs["paths_end"],   N_PATHS, 3),                        # N*3
        _pad_sequence(obs["paths_width"], N_PATHS, 1),                        # N*1
    ])


def decode_action(action: np.ndarray) -> dict:
    """Map a 5-dim float array in [-1, 1] to an STK action dict."""
    return {
        "acceleration": np.array([(float(action[0]) + 1.0) / 2.0]),
        "steer":        np.array([float(action[1])]),
        "brake":        int(action[2] > 0),
        "drift":        int(action[3] > 0),
        "nitro":        int(action[4] > 0),
    }


def expert_action_to_flat(action: dict) -> np.ndarray:
    """Convert an STK dict action to a 5-dim float32 array in [-1, 1]."""
    accel = float(np.asarray(action["acceleration"]).flat[0])
    steer = float(np.asarray(action["steer"]).flat[0])
    return np.array([
        2.0 * accel - 1.0,
        steer,
        1.0 if action.get("brake", False) else -1.0,
        1.0 if action.get("drift", False) else -1.0,
        1.0 if action.get("nitro", False) else -1.0,
    ], dtype=np.float32)


# ---------------------------------------------------------------------------
# STKFlatWrapper
# ---------------------------------------------------------------------------

class STKFlatWrapper(gym.Wrapper):
    """Flat float32 obs (OBS_DIM,) + 5-dim continuous action on full-v0."""

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(OBS_DIM,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(5,), dtype=np.float32
        )

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return obs_to_flat(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(decode_action(action))
        return obs_to_flat(obs), reward, terminated, truncated, info


def make_stk_flat_env(track: str | None = None, num_kart: int = 1) -> gym.Env:
    """Create a flat STK env ready for SAC/RLPD training."""
    import pystk2_gymnasium  # noqa: F401 — registers envs
    env = gym.make("supertuxkart/full-v0", track=track, num_kart=num_kart, render_mode=None)
    return STKFlatWrapper(env)


# ---------------------------------------------------------------------------
# STKVecEnv — rl_mind VecEnv interface, single STK instance
# ---------------------------------------------------------------------------

class STKVecEnv:
    """Single-instance STK environment with the rl_mind VecEnv interface.

    pystk2 is a per-process singleton, so num_envs is always 1.
    Same-step reset: when an episode ends, next_obs holds the terminal obs
    and obs holds the first obs of the new episode.
    """

    num_envs: int = 1
    same_step_reset: bool = True

    def __init__(self, track: str | None = None, num_kart: int = 1, seed: int | None = None):
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
        obs, _ = self._env.reset(seed=self._seed)
        self._seed = None
        return torch.from_numpy(obs).unsqueeze(0)

    def step(self, actions: Tensor) -> EnvStep:
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
# Expert data collection (RLPD offline buffer)
# ---------------------------------------------------------------------------

def collect_expert_data(
    agent_cls,
    n_steps: int,
    track: str | None = None,
    num_kart: int = 1,
    **agent_kwargs,
) -> ReplayBuffer:
    """Run a KartAgent expert for n_steps, return a populated ReplayBuffer.

    The expert receives raw full-v0 Dict observations (cartesian coords,
    variable-length sequences) which is what the expert agents expect.
    The buffer stores flat obs and 5-dim actions matching the training env,
    so it can be passed directly to run_rlpd() as offline_buffer.
    """
    import pystk2_gymnasium  # noqa: F401 — registers envs

    env = gym.make("supertuxkart/full-v0", track=track, num_kart=num_kart, render_mode=None)
    expert = agent_cls(env, **agent_kwargs)
    buffer = ReplayBuffer(n_steps)

    obs_dict, _ = env.reset()
    expert.obs = obs_dict
    expert.agent_positions = []

    collected = 0
    while collected < n_steps:
        flat_obs = obs_to_flat(obs_dict)
        action_dict = expert.choose_action(obs_dict)
        flat_action = expert_action_to_flat(action_dict)

        next_obs_dict, reward, terminated, truncated, _ = env.step(action_dict)

        buffer.add(Transitions(
            obs=torch.from_numpy(flat_obs).unsqueeze(0),
            action=Action(value=torch.from_numpy(flat_action).unsqueeze(0)),
            reward=torch.tensor([float(reward)], dtype=torch.float32),
            next_obs=torch.from_numpy(obs_to_flat(next_obs_dict)).unsqueeze(0),
            terminated=torch.tensor([terminated], dtype=torch.bool),
        ))
        collected += 1

        obs_dict = next_obs_dict
        if terminated or truncated:
            obs_dict, _ = env.reset()
            expert.obs = obs_dict
            expert.agent_positions = []

    env.close()
    return buffer
