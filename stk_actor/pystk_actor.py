"""Race actor for the CLI runner (PYTHONPATH=. pystk2 race --hide stk_actor).

Switch between EXPERT and NETWORK by changing the MODE variable below.
"""

import sys
from pathlib import Path
from typing import Callable, List

import gymnasium as gym

# ---------------------------------------------------------------------------
# MODE: "expert" to run a rule-based expert agent, "network" to run the
# trained SAC/RLPD/DAgger actor loaded from pystk_actor.pth.
# ---------------------------------------------------------------------------
MODE = "expert"
EXPERT_NAME = "team5"   # team1 .. team5

# Full-v0 gives Dict obs in cartesian coords — what the expert agents expect.
# Our STKFlatWrapper is added in get_wrappers() when MODE == "network".
env_name = "supertuxkart/full-v0"
player_name = "Ni20"


def get_wrappers() -> List[Callable[[gym.Env], gym.Wrapper]]:
    # No wrapper: pystk2 race only accepts observation/action wrappers, so the
    # network actor flattens the obs and decodes its action itself
    return []


def create_state():
    return None


def get_actor(
    module_dir: Path,
    observation_space: gym.spaces.Space,
    action_space: gym.spaces.Space,
) -> Callable:
    if MODE == "expert":
        return _make_expert_actor(module_dir)
    else:
        return _make_network_actor(module_dir)


# ---------------------------------------------------------------------------
# Expert actor
# ---------------------------------------------------------------------------

def _make_expert_actor(module_dir: Path) -> Callable:
    """Load the requested expert agent and return an actor callable.

    The CLI runner calls actor(state, obs) with a Dict obs from full-v0.
    Expert agents only need that Dict in choose_action(); the env reference
    they store is used for world-state logging (KartAgent.step), not for
    action selection.  We pass a lightweight stub so the constructor succeeds.
    """
    # Ensure expert/ is on the path (pre-import pystk2_gymnasium first so the
    # local expert/pystk2_gymnasium/ folder does not shadow the venv copy).
    import pystk2_gymnasium  # noqa: F401 — cache installed version first
    expert_dir = str(module_dir.parent / "expert")
    if expert_dir not in sys.path:
        sys.path.insert(0, expert_dir)

    _table = {
        "team1": ("agents.team1.agent1", "Agent1"),
        "team2": ("agents.team2.agent2", "Agent2"),
        "team3": ("agents.team3.agent3", "Agent3"),
        "team4": ("agents.team4.agent4", "Agent4"),
        "team5": ("agents.team5.agent5", "Agent5"),
    }
    module_name, cls_name = _table[EXPERT_NAME]
    import importlib
    cls = getattr(importlib.import_module(module_name), cls_name)

    # Stub env: satisfies the constructor without a live pystk2 instance.
    # (Nested classes can't reference their outer class, so define flat.)
    class _Kart:
        location = [0.0, 0.0, 0.0]
    class _World:
        karts = [_Kart()]
    class _Unwrapped:
        world = _World()
    class _StubEnv:
        unwrapped = _Unwrapped()

    expert = cls(_StubEnv())

    def actor(state, obs):
        return expert.choose_action(obs)

    return actor


# ---------------------------------------------------------------------------
# Network actor
# ---------------------------------------------------------------------------

def _make_network_actor(module_dir: Path) -> Callable:
    """Load the saved SAC/RLPD/DAgger actor from pystk_actor.pth."""
    import torch
    from .sac import SquashedGaussianActor
    from .stk_bridge import OBS_DIM

    weights_path = module_dir / "pystk_actor.pth"
    action_dim = 6
    actor = SquashedGaussianActor(OBS_DIM, (256, 256), action_dim)
    actor.load_state_dict(torch.load(weights_path, map_location="cpu"))
    actor.eval()

    def act(state, obs):
        import numpy as np
        from .stk_bridge import obs_to_flat
        obs_t = torch.from_numpy(obs_to_flat(obs)).unsqueeze(0)
        with torch.no_grad():
            action_t = actor.act(obs_t)
        from .stk_bridge import decode_action
        return decode_action(action_t[0].numpy())

    return act
