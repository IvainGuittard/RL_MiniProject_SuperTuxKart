"""Bridge between pystk2-gymnasium and the rl_mind SAC / RLPD / DAgger pipeline.

Uses supertuxkart/full-v0 directly — no ConstantSizedObservations or
PolarObservations dependency.  Variable-length path sequences are padded here,
so num_kart=1 works without hitting the installed-wrapper np.stack crash.

Observation (flat float32, OBS_DIM = 11 + N_PATHS * 7), kart-centric frame:
  velocity              3   kart velocity
  center_path_distance  1   lateral offset from track centre
  front                 3   kart heading direction
  center_path           3   nearest track-centre position
  energy                1   remaining nitro energy
  paths_start      N×3     upcoming track-node start positions (padded)
  paths_end        N×3     upcoming track-node end positions
  paths_width      N×1     upcoming track-node widths

distance_down_track is deliberately left out: absolute progress in metres
means something different on every track and does not help steering.

Learned action (ACTION_DIM = 4, all in [-1, 1]):
  0  acceleration   mapped to [0, 1] by (x+1)/2
  1  steer          kept in [-1, 1]
  2  drift          x > 0 → 1, else 0
  3  nitro          x > 0 → 1, else 0

brake and rescue are always off for the network (SAC kept pressing them at
random, which froze the kart). The experts are NOT restricted: they execute
their own full actions (brake / rescue included, see expert_to_env), and the
dataset stores the projection of those actions onto the 4 learned dims.
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
ACTION_DIM = 4                 # accel, steer, drift, nitro


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
        np.asarray(obs["center_path_distance"], dtype=np.float32).flatten(),  # 1
        np.asarray(obs["front"],                dtype=np.float32).flatten(),  # 3
        np.asarray(obs["center_path"],          dtype=np.float32).flatten(),  # 3
        np.asarray(obs["energy"],               dtype=np.float32).flatten(),  # 1
        _pad_sequence(obs["paths_start"], N_PATHS, 3),                        # N*3
        _pad_sequence(obs["paths_end"],   N_PATHS, 3),                        # N*3
        _pad_sequence(obs["paths_width"], N_PATHS, 1),                        # N*1
    ])


def decode_action(action) -> dict:
    """Map an ACTION_DIM float array in [-1, 1] to an STK action dict.

    A dict (an expert action from expert_to_env) is passed through as-is.
    """
    if isinstance(action, dict):
        return action
    return {
        "acceleration": np.array([(float(action[0]) + 1.0) / 2.0]),
        "steer":        np.array([float(action[1])]),
        "brake":        0,
        "drift":        int(action[2] > 0),
        "nitro":        int(action[3] > 0),
        "rescue":       0,
        "fire":         0,
    }


def expert_to_env(action: dict, accel: float | None = None, steer: float | None = None) -> dict:
    """The expert's own full action, as executed in the env (fire off, as
    before). ``accel`` / ``steer`` override those two values (exploration
    noise during data collection)."""
    if accel is None:
        accel = float(np.clip(np.asarray(action["acceleration"]).flat[0], 0.0, 1.0))
    if steer is None:
        steer = float(np.clip(np.asarray(action["steer"]).flat[0], -1.0, 1.0))
    return {
        "acceleration": np.array([accel]),
        "steer":        np.array([steer]),
        "brake":        int(bool(action.get("brake", False))),
        "drift":        int(bool(action.get("drift", False))),
        "nitro":        int(bool(action.get("nitro", False))),
        "rescue":       int(bool(action.get("rescue", False))),
        "fire":         0,
    }


def expert_action_to_flat(action: dict) -> np.ndarray:
    """Project an STK dict action onto the ACTION_DIM learned dims, in [-1, 1]
    (brake / rescue are not part of the learned action)."""
    accel = float(np.clip(np.asarray(action["acceleration"]).flat[0], 0.0, 1.0))
    steer = float(np.clip(np.asarray(action["steer"]).flat[0], -1.0, 1.0))
    return np.array([
        2.0 * accel - 1.0,
        steer,
        1.0 if action.get("drift", False) else -1.0,
        1.0 if action.get("nitro", False) else -1.0,
    ], dtype=np.float32)


# ---------------------------------------------------------------------------
# STKFlatWrapper
# ---------------------------------------------------------------------------

class STKFlatWrapper(gym.Wrapper):
    """Flat float32 obs (OBS_DIM,) + ACTION_DIM continuous action on full-v0."""

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(OBS_DIM,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(ACTION_DIM,), dtype=np.float32
        )
        self.raw_obs: dict | None = None

    @property
    def track_length(self) -> float:
        """Length of the current track (available after reset)."""
        return float(self.env.unwrapped.track.length)

    @property
    def max_episode_steps(self) -> int:
        return int(self.env.spec.max_episode_steps)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.raw_obs = obs
        return obs_to_flat(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(decode_action(action))
        self.raw_obs = obs
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
    def flat_env(self) -> STKFlatWrapper:
        """The underlying (single) gymnasium env, shared by training,
        expert data collection and evaluation (pystk2 is a singleton)."""
        return self._env

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
# Expert dataset (shared by RLPD and DAgger)
# ---------------------------------------------------------------------------

def run_expert_lap(env: STKFlatWrapper, expert_cls) -> dict:
    """Run the expert without noise for one episode; return its lap stats.

    The expert executes its own full action (fire off).
    """
    env.reset()
    expert = expert_cls(env)
    steps, ret, done = 0, 0.0, False
    terminated = False
    while not done:
        _, reward, terminated, truncated, _ = env.step(
            expert_to_env(expert.choose_action(env.raw_obs)))
        steps += 1
        ret += float(reward)
        done = terminated or truncated
    return {"finished": bool(terminated), "steps": steps, "return": ret}


def collect_expert_dataset(
    env: STKFlatWrapper,
    expert_cls,
    n_steps: int,
    noise_std: float = 0.2,
    seed: int = 0,
) -> dict:
    """Run the expert for ``n_steps`` steps and return the dataset.

    Since the environment is deterministic, a noiseless expert would replay
    the same lap over and over. Gaussian noise of std ``noise_std`` is
    therefore added to the executed acceleration and steering, to cover the
    states around the expert trajectory.

    Two actions are stored for every state:

    - ``action``: the action actually executed (noisy), projected onto the
      learned dims, which produced ``reward`` / ``next_obs`` (what RLPD's
      offline buffer needs);
    - ``expert_action``: the clean expert action in that state (the label
      DAgger / behaviour cloning learns from).

    :returns: A dict of tensors (``obs``, ``action``, ``expert_action``,
        ``reward``, ``next_obs``, ``terminated``) and a ``meta`` dict.
    """
    rng = np.random.default_rng(seed)
    obs_l, act_l, exp_l, rew_l, next_l, term_l = [], [], [], [], [], []
    episodes = []

    obs, _ = env.reset()
    expert = expert_cls(env)
    ep_steps, ep_return = 0, 0.0
    for _ in range(n_steps):
        raw = expert.choose_action(env.raw_obs)
        expert_action = expert_action_to_flat(raw)
        action = expert_action.copy()
        if noise_std > 0:
            action[:2] = np.clip(action[:2] + rng.normal(0.0, noise_std, 2), -1.0, 1.0)

        # The expert's full action is executed (brake / rescue included),
        # with the noisy acceleration and steering
        next_obs, reward, terminated, truncated, _ = env.step(
            expert_to_env(raw, accel=(float(action[0]) + 1.0) / 2.0, steer=float(action[1])))
        obs_l.append(obs)
        act_l.append(action)
        exp_l.append(expert_action)
        rew_l.append(float(reward))
        next_l.append(next_obs)
        term_l.append(bool(terminated))
        ep_steps += 1
        ep_return += float(reward)

        obs = next_obs
        if terminated or truncated:
            episodes.append({"finished": bool(terminated), "steps": ep_steps, "return": ep_return})
            obs, _ = env.reset()
            expert = expert_cls(env)
            ep_steps, ep_return = 0, 0.0

    return {
        "obs": torch.from_numpy(np.stack(obs_l)),
        "action": torch.from_numpy(np.stack(act_l)),
        "expert_action": torch.from_numpy(np.stack(exp_l)),
        "reward": torch.tensor(rew_l, dtype=torch.float32),
        "next_obs": torch.from_numpy(np.stack(next_l)),
        "terminated": torch.tensor(term_l, dtype=torch.bool),
        "meta": {
            "n_steps": n_steps,
            "noise_std": noise_std,
            "seed": seed,
            "episodes": episodes,
        },
    }


def save_dataset(dataset: dict, path) -> None:
    from pathlib import Path
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(dataset, path)


def load_dataset(path) -> dict:
    return torch.load(path, weights_only=False)


def dataset_to_buffer(dataset: dict) -> ReplayBuffer:
    """The offline ReplayBuffer of RLPD (executed actions and rewards)."""
    buffer = ReplayBuffer(len(dataset["obs"]))
    buffer.add(Transitions(
        obs=dataset["obs"],
        action=Action(value=dataset["action"]),
        reward=dataset["reward"],
        next_obs=dataset["next_obs"],
        terminated=dataset["terminated"],
    ))
    return buffer
