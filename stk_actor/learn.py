"""Training script example.

Run from the repository root:

    PYTHONPATH=. python -m stk_actor.learn
"""

from pathlib import Path

import gymnasium as gym
from pystk2_gymnasium import AgentSpec

# Note the use of relative imports
from .pystk_actor import env_name, get_wrappers, player_name


if __name__ == "__main__":
    # (1) Create the environment
    env = gym.make(
        env_name,
        render_mode=None,
        agent=AgentSpec(use_ai=False, name=player_name),
    )
    for wrapper_fn in get_wrappers():
        env = wrapper_fn(env)

    # (2) Train your agent
    # ...

    # (3) Save your model weights
    mod_path = Path(__file__).parent
    # Example: torch.save(model.state_dict(), mod_path / "pystk_actor.pth")
