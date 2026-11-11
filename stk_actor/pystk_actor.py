from pathlib import Path
from typing import Callable, List

import gymnasium as gym

# IMPORTANT: use relative imports within your module
from .actors import MyWrapper, make_random_actor

#: The base environment name (you can change that)
env_name = "supertuxkart/flattened_multidiscrete-v0"

#: Player name (you must change that)
player_name = "Example"


def get_wrappers() -> List[Callable[[gym.Env], gym.Wrapper]]:
    """Returns a list of additional wrappers to be applied to the base
    environment."""
    return [
        # Example of a custom wrapper
        lambda env: MyWrapper(env, option="1")
    ]


# --- Agent state ---
#
# ``create_state()`` is called once at the start of each race. The returned
# value is passed as the *first* argument to the actor on every step.
#
# Return ``None`` if your agent does not need persistent state (stateless).
# Return a dict (or any object) if your agent needs to track information
# across steps (e.g. a recurrent model, a history buffer, or a step counter).


def create_state():
    """Return a fresh state object for a new race.

    Called once at the beginning of each race. The returned value is
    passed as the first argument to the actor callable on every step.

    Return ``None`` for a stateless agent, or a dict/object for a
    stateful one. Examples::

        # Stateless:
        return None

        # Stateful:
        return {"step": 0, "history": []}
    """
    return None


def get_actor(
    module_dir: Path,
    observation_space: gym.spaces.Space,
    action_space: gym.spaces.Space,
) -> Callable:
    """Create an actor callable for gameplay.

    :param module_dir: Path to the directory containing this file
        (use it to load weights, e.g. ``module_dir / "pystk_actor.pth"``)
    :param observation_space: The (wrapped) observation space
    :param action_space: The (wrapped) action space
    :return: A callable ``actor(state, obs) -> action``. The *state*
        argument is whatever ``create_state()`` returned (may be ``None``).
    """

    # --- Stateless example (create_state returns None) ---
    return make_random_actor(observation_space, action_space)

    # --- Stateful example (change create_state to return a dict) ---
    # return make_stateful_actor(module_dir, observation_space, action_space)
