import gymnasium as gym


class MyWrapper(gym.ActionWrapper):
    """Example custom wrapper (does nothing)."""

    def __init__(self, env, option: int):
        super().__init__(env)
        self.option = option

    def action(self, action):
        return action


def make_random_actor(observation_space, action_space):
    """Return a stateless actor that samples random actions."""

    def actor(state, obs):
        return action_space.sample()

    return actor


def make_stateful_actor(module_dir, observation_space, action_space):
    """Return a stateful actor that counts steps.

    All actors receive ``(state, obs)``. The state object is created by
    ``create_state()`` (see pystk_actor.py) and persists across steps
    within a single race. The actor can mutate it in place.
    """
    # Example: load trained weights from module_dir
    # weights_path = Path(module_dir) / "pystk_actor.pth"
    # model = ...
    # model.load_state_dict(torch.load(weights_path))

    def actor(state, obs):
        state["step"] += 1

        # Replace with your own logic
        return action_space.sample()

    return actor
