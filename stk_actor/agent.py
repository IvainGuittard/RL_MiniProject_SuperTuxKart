"""SuperTuxKart agent shim: upload this file unchanged, next to your own files.

The platform's agent contract is a zero-argument ``class Agent`` in ``agent.py``.
This shim turns it into the upstream pystk2-gymnasium agent format, the one
``pystk2 race`` accepts, so your ``stk_actor/`` code runs as it does locally:

- Upload the *files* of your ``stk_actor/`` folder, flat, next to this
  ``agent.py``: ``pystk_actor.py`` (required), ``pystk_actor.pth`` and your
  sibling modules (``actors.py``, ...). ``__init__.py`` is not needed: the shim
  makes the upload directory the ``stk_actor`` package, so ``from .actors
  import ...`` works.
- ``pystk_actor.py`` defines, as upstream: ``env_name`` (default
  ``supertuxkart/full-v0``), ``player_name``, ``get_wrappers()`` (optional),
  ``create_state()`` (optional) and ``get_actor(module_dir, obs_space,
  act_space) -> actor(state, obs)``.

What env.py calls (CONTRACT.md §B):

- ``setup(info) -> dict`` once per race, before the first step;
- ``act(obs) -> dict`` once per step: ``obs`` is this kart's base
  ``supertuxkart/multi-full-v0`` observation, the return value the base action.

The observation goes through your ``env_name`` wrappers and ``get_wrappers()``
here, inside the agent container, with upstream ``race_server``'s
``_build_wrapper_adapter`` (``MonoAgentWrapperAdapter`` over a
``_FakeMultiAgentEnv`` of the base spaces, rebuilt locally), and the actor's
action is unwrapped back to the base action. There is no error handling on
purpose: any exception reaches the platform, which records it as your agent's
code error with the traceback.
"""
import importlib
import importlib.machinery
import importlib.util
import math
import sys
from pathlib import Path

PROTOCOL = 2
ENV_ID = "supertuxkart/multi-full-v0"
PACKAGE = "stk_actor"
CONTINUOUS_ACTIONS = ("acceleration", "steer")
DISCRETE_ACTIONS = ("brake", "drift", "fire", "nitro", "rescue")


class Agent:
    def __init__(self):
        # Stdlib only: Hello has a short deadline. torch / pystk2 load in setup().
        self.state = None
        self._actor = None
        self._adapter = None
        self._sequence_keys = ()

    def setup(self, info):
        if self._actor is not None:
            raise RuntimeError("setup() was already called for this race")
        if info["protocol"] != PROTOCOL:
            raise ValueError(f"unsupported protocol {info['protocol']!r} (this shim speaks {PROTOCOL})")
        if info["env_id"] != ENV_ID:
            raise ValueError(f"unsupported env {info['env_id']!r} (expected {ENV_ID!r})")
        if importlib.util.find_spec("pystk2_gymnasium") is None:
            raise ModuleNotFoundError(
                "pystk2_gymnasium is not installed in this agent runtime: pick the "
                "SuperTuxKart runtime (PyTorch · SuperTuxKart) for this submission"
            )

        import pystk2_gymnasium
        from gymnasium import spaces
        from pystk2_gymnasium.cli.race import LoadedAgent
        from pystk2_gymnasium.cli.race_server import _build_wrapper_adapter
        from pystk2_gymnasium.envs import kart_action_space, kart_observation_space

        if pystk2_gymnasium.__version__ != info["pystk2_gymnasium_version"]:
            raise ValueError(
                f"pystk2-gymnasium {pystk2_gymnasium.__version__} in the agent image, "
                f"{info['pystk2_gymnasium_version']} in the env: pick the matching agent runtime"
            )

        module_dir = Path(__file__).resolve().parent
        module = _import_participant(module_dir)

        # The same defaults as upstream pystk2_gymnasium.cli.race.load_agent.
        loaded = LoadedAgent(
            env_name=getattr(module, "env_name", "supertuxkart/full-v0"),
            player_name=getattr(module, "player_name", ""),
            get_actor=module.get_actor,
            module_dir=module_dir,
            get_wrappers=getattr(module, "get_wrappers", None),
            source=str(module_dir),
            create_state=getattr(module, "create_state", lambda: None),
        )

        obs_space = kart_observation_space(False)
        self._sequence_keys = tuple(
            key for key, space in obs_space.spaces.items() if isinstance(space, spaces.Sequence)
        )
        # Upstream race_server's own server-side wrapping: MonoAgentWrapperAdapter
        # over a _FakeMultiAgentEnv of the base spaces, one agent under key "0".
        self._adapter = _build_wrapper_adapter(
            ["0"], {"0": 0}, [loaded], {"0": obs_space}, {"0": kart_action_space()}
        )
        wrapped_obs_space = self._adapter.observation_space["0"]
        wrapped_act_space = self._adapter.action_space["0"]
        self._actor = loaded.get_actor(loaded.module_dir, wrapped_obs_space, wrapped_act_space)
        self.state = loaded.create_state()

        return {
            "protocol": PROTOCOL,
            "player_name": str(loaded.player_name),
            "env_name": str(loaded.env_name),
            "pystk2_gymnasium_version": pystk2_gymnasium.__version__,
        }

    def act(self, obs):
        if self._actor is None:
            raise RuntimeError("act() before setup()")
        from pystk2_gymnasium.envs import _scalar

        # The wire carries a Sequence as a list; gymnasium's Sequence space and
        # the wrappers expect the tuple the env produced.
        for key in self._sequence_keys:
            obs[key] = tuple(obs[key])
        wrapped = self._adapter.observation({"0": obs})["0"]
        action = self._actor(self.state, wrapped)
        base = self._adapter.action({"0": action})["0"]

        expected = set(CONTINUOUS_ACTIONS + DISCRETE_ACTIONS)
        if not isinstance(base, dict) or set(base) != expected:
            keys = sorted(base) if isinstance(base, dict) else type(base).__name__
            raise ValueError(f"the base action must have exactly the keys {sorted(expected)}, got {keys}")
        # _scalar is what upstream's env applies to each action value: .item()
        # refuses an array of more than one element.
        out = {key: float(_scalar(base[key])) for key in CONTINUOUS_ACTIONS}
        for key, value in out.items():
            if not math.isfinite(value):
                raise ValueError(f"action {key!r} is not a finite number: {value!r}")
        out.update({key: int(_scalar(base[key])) for key in DISCRETE_ACTIONS})
        return out


def _import_participant(module_dir):
    """Import ``stk_actor.pystk_actor`` from the flat upload directory.

    The directory is read-only and may hold no ``__init__.py`` (an upload stores
    it as ``init__.py``), so the ``stk_actor`` package is registered in memory
    with the directory as its path: relative imports (``from .actors import``)
    resolve to the uploaded siblings, and nothing is written.
    """
    if not (module_dir / "pystk_actor.py").is_file():
        raise FileNotFoundError(
            f"no pystk_actor.py next to agent.py in {module_dir}: upload the files of your "
            "stk_actor/ folder (flat) together with this agent.py"
        )
    package = sys.modules.get(PACKAGE)
    if package is None:
        spec = importlib.machinery.ModuleSpec(PACKAGE, None, is_package=True)
        spec.submodule_search_locations = [str(module_dir)]
        package = importlib.util.module_from_spec(spec)
        sys.modules[PACKAGE] = package
    elif list(package.__path__) != [str(module_dir)]:
        raise ImportError(f"a different {PACKAGE!r} package is already imported: {list(package.__path__)}")
    return importlib.import_module(f"{PACKAGE}.pystk_actor")

