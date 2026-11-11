# PySTK2-gymnasium project template

This project template contains a basic structure for a PySTK2 reinforcement learning agent.
For information about the PySTK2 gymnasium environment, please look at the [corresponding github page](https://github.com/bpiwowar/pystk2-gymnasium)

## Setup

### On PPTI machines

A Python environment with all the required libraries is already installed:
**do not create your own**, just activate it (in every new terminal):

```sh
source PPTI_ENV_PATH/bin/activate
```

and define the `PYSTK2` and `PYTHON` aliases used in the commands below:

```sh
alias PYSTK2=pystk2
alias PYTHON=python
```

### On your own computer

We use [uv](https://docs.astral.sh/uv/) to manage the Python environment
(see [installation instructions](https://docs.astral.sh/uv/getting-started/installation/)).

```sh
# To be run from the base directory: creates a virtual environment in .venv
uv venv
# Install the PySTK2 gymnasium environment with the race runner (cli) and
# the web dashboard (web) -- add any other library you need (but only the
# libraries available on ML-Arena can be used, see below)
uv pip install "pystk2-gymnasium[cli,web]==0.10.0"
```

Then define the `PYSTK2` and `PYTHON` aliases used in the commands below (in every new
terminal); `uv run` runs the command within this environment:

```sh
alias PYSTK2="uv run pystk2"
alias PYTHON="uv run python"
```

## Structure

**Warning**: all the imports should be relative within your module (see `learn.py` for an example).

### `agent.py`

The [ML-Arena](https://ml-arena.com/viewchallenge/169) shim that runs your agent
on the platform: **do not modify it** (and you do not need to read it).

### `actors.py`

Contains the actors used throughout your project

### `learn.py`

Contains the code to train your actor

### `pystk_actor.py`

This Python file (**don't change its name**) should contain:

- `env_name`: The base environment name
- `player_name`: The actor name (displayed on top of the kart)
- `get_wrappers()` *(optional)*: Returns a list of extra wrappers applied to the environment
- `get_actor(module_dir, observation_space, action_space)`: Returns a callable `actor(state, obs) -> action`. The `module_dir` parameter is the path to the directory containing this file (use it to load saved weights, e.g. `module_dir / "pystk_actor.pth"`)
- `create_state()` *(optional)*: Called once per race to create a fresh state object; the state is then passed as the first argument to the actor on every step. Return `None` for stateless agents (the default if not defined).

### Unified actor interface

All actors use the same signature: `actor(state, obs) -> action`.

The `create_state()` function determines whether the agent carries state:

**Stateless** (`create_state` returns `None`):

```python
# pystk_actor.py
def create_state():
    return None

def get_actor(module_dir, observation_space, action_space):
    def actor(state, obs):  # state is None
        return action_space.sample()
    return actor
```

**Stateful** (`create_state` returns a dict or any object):

```python
# pystk_actor.py
def create_state():
    return {"step": 0, "history": []}

def get_actor(module_dir, observation_space, action_space):
    def actor(state, obs):
        state["step"] += 1
        state["history"].append(obs)
        return action_space.sample()
    return actor
```

## Learn your model

```sh
# To be run from the base directory
PYTHONPATH=. PYTHON -m stk_actor.learn
```

This should create the `pystk_actor.pth` file (**don't change its name**) that contains the parameters of your model. Load it in `get_actor` using `module_dir / "pystk_actor.pth"`.

## Testing the actor

To test your agent locally:

```sh
PYTHONPATH=. PYSTK2 race --hide stk_actor
```

You can also compare different actors in the same race:

```sh
PYTHONPATH=. PYSTK2 race --hide stk_actor other_actor
```

### Interacting with the race (web dashboard)

To see what your agent is doing, use the `--web` option: it starts a web
server displaying the track map (3D view with the karts positions) and
information about each kart (observations, rewards).

```sh
PYTHONPATH=. PYSTK2 race --hide --web stk_actor
```

Then open <http://localhost:8050> in your browser (use `--web-port PORT` to
change the port, e.g. if several people share the same machine). The race
starts **paused**; use the buttons to control it:

- **Step**: advance the race by one step
- **Run** / **Pause**: run the race continuously / pause it
- **Stop**: end the race

Other useful options: `--track TRACK` (choose the track), `--num-karts N`,
`--laps N`, `--max-steps N`, and `--record race.mp4` to save a video of the race
(see `PYSTK2 race --help` for the full list).

## Submit your work on ML-Arena

Your agent is evaluated on [ML-Arena](https://ml-arena.com/viewchallenge/169):
four karts race on a random track, and agents are ranked by a pairwise Elo
rating (a race gives 4 - placement points).

1. **Check your agent locally**

   ```sh
   uv run scripts/check_submission.py
   ```

   This runs with the libraries of the platform (the first run downloads
   them): it checks the files that will be uploaded (`agent.py`,
   `pystk_actor.py`, `pystk_actor.pth`, `actors.py`, ... — `__init__.py` and
   `learn.py` are left out) and the upload limits, then loads your agent the
   way the platform does and plays a short race (use `--steps N` and
   `--track TRACK` to change it).

2. **Submit it**, either from the command line or from the web site.

   *From the command line*: create an API key on your ML-Arena profile
   (Profile → API Keys) and put it in a `.env` file in the base directory
   (it is ignored by git: **never commit it**)

   ```sh
   MLARENA_API_KEY=mlk_...
   ```

   then run

   ```sh
   uv run scripts/submit.py
   ```

   The submission is named after `player_name` (in `pystk_actor.py`): change
   it from `"Example"` first (or use `--name NAME`). This uploads the files of
   `stk_actor/`, selects the right runtime, deploys the submission and
   follows its evaluation (use `--no-wait` to return immediately).

   A deployed submission cannot be modified on ML-Arena, so each call
   creates a new submission (at most 20 active submissions and 5 deployments
   per day). Its id is saved in `.mlarena_submission`, so that the next call
   updates it instead if it could not be deployed (use `--new` to always
   create a new one, or `--submission-id ID` to update a given one).

   *From the web site*: upload the files of the `stk_actor/` folder (not the
   folder itself, and without `__init__.py` and `learn.py`) on the [challenge
   page](https://ml-arena.com/viewchallenge/169), and choose the runtime
   *PyTorch · SuperTuxKart (pystk2-gymnasium, SB3)* (with any other runtime,
   your agent fails).

3. **Check the runs** of your submission: if your agent fails, the error and
   its traceback are shown there.

Platform constraints:

- **Files**: at most 10 files and 100 MB in total, all in one folder (no
  sub-folders, so all your modules must be next to `pystk_actor.py`)
- **Libraries**: only those of the runtime (Python 3.12, `pystk2-gymnasium`
  0.10.0, PyTorch 2.12, Stable-Baselines3 2.9 and SB3-Contrib 2.9, gymnasium,
  numpy, pandas, matplotlib...): you cannot install other libraries
- **Resources**: CPU only (no GPU), 3 CPUs, 3 GB of memory and **at most 1
  second per step**
