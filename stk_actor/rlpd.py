"""RLPD: Reinforcement Learning with Prior Data (Ball et al., 2023).

Extends SAC with:
  - Two replay buffers: a frozen offline buffer (expert data) and a growing
    online buffer; each gradient step draws half the batch from each.
  - Ensemble of E critics, with random subset Z used for target computation
    (Random Ensemble Distillation, RED).
  - Optional LayerNorm in critics, which stabilises Q-learning on mixed data.

The offline buffer is passed in already populated (see collect_expert_dataset()
in stk_actor/stk_bridge.py).

Run from the repository root via learn.py:

    PYTHONPATH=. python -m stk_actor.learn rlpd --expert team3 --utd 10
"""

import copy
import math
import random
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

from rl_mind.nn import soft_update
from rl_mind.env import VecEnv
from rl_mind.data import ReplayBuffer, Transitions
from rl_mind.collectors import TransitionCollector
from rl_mind.evaluation import Evaluator

from .sac import SACConfig, SquashedGaussianActor


# ---------------------------------------------------------------------------
# Critic ensemble (vectorised: E critics in one batched matmul per layer)
# ---------------------------------------------------------------------------

class EnsembleQNetwork(nn.Module):
    """E independent Q(s, a) MLPs evaluated in parallel.

    Each member has its own weights (shape [E, d_in, d_out]) and, optionally,
    its own LayerNorm after each hidden layer (RLPD paper §4).
    forward() returns Q-values of shape [E, B].
    """

    def __init__(
        self,
        n: int,
        obs_dim: int,
        hidden: tuple[int, ...],
        action_dim: int,
        layer_norm: bool = True,
    ):
        super().__init__()
        self.n = n
        self.layer_norm = layer_norm
        dims = [obs_dim + action_dim, *hidden, 1]
        self.weights = nn.ParameterList()
        self.biases = nn.ParameterList()
        for d_in, d_out in zip(dims[:-1], dims[1:]):
            # Same init as nn.Linear: U(-1/sqrt(d_in), 1/sqrt(d_in))
            bound = 1.0 / math.sqrt(d_in)
            self.weights.append(nn.Parameter(torch.empty(n, d_in, d_out).uniform_(-bound, bound)))
            self.biases.append(nn.Parameter(torch.empty(n, 1, d_out).uniform_(-bound, bound)))
        if layer_norm:
            self.ln_scales = nn.ParameterList(nn.Parameter(torch.ones(n, 1, d)) for d in hidden)
            self.ln_shifts = nn.ParameterList(nn.Parameter(torch.zeros(n, 1, d)) for d in hidden)

    def forward(self, obs: Tensor, action: Tensor) -> Tensor:
        x = torch.cat([obs, action], dim=1).unsqueeze(0).expand(self.n, -1, -1)
        last = len(self.weights) - 1
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            x = torch.baddbmm(b, x, w)
            if i < last:
                if self.layer_norm:
                    x = F.layer_norm(x, x.shape[-1:]) * self.ln_scales[i] + self.ln_shifts[i]
                x = F.relu(x)
        return x.squeeze(-1)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RLPDConfig(SACConfig):
    # --- offline buffer ---
    # Fraction of each batch drawn from the offline (expert) buffer.
    offline_fraction: float = 0.5
    # RLPD can start training as soon as the online buffer has one half-batch.
    learning_starts: int = 256

    # --- update-to-data ratio ---
    # The UTD ratio is updates_per_round / steps_per_round (gradient steps per
    # env step). RLPD relies on a high UTD (paper: 20); 32/32 or 1/1 is UTD 1.
    steps_per_round: int = 1
    updates_per_round: int = 10

    # Paper value (SACConfig's 0.05 is 10x higher, too fast with a high UTD)
    tau: float = 0.005

    # --- critic ensemble (Random Ensemble Distillation) ---
    # E in the paper: number of critics to train.
    n_critics: int = 10
    # Z in the paper: size of the random subset used to compute the target.
    critic_subset_size: int = 2

    # --- ablation flags ---
    use_layer_norm: bool = True   # LayerNorm in critics (key RLPD contribution)
    # entropy_mode inherited from SACConfig: "auto" or "fixed"

    # Rewards are multiplied by this in the Bellman target (the env reward
    # is ~0.1/step, small next to SAC's entropy bonus). Evaluation still
    # reports the raw env return; Q-values are in scaled units.
    reward_scale: float = 1.0

    # TensorBoard logging period for the losses (in env steps)
    log_interval: int = 1_000


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def run_rlpd(
    cfg: RLPDConfig,
    offline_buffer: ReplayBuffer,
    run_dir: str | None = None,
    stk_env=None,
    make_evaluator=None,
):
    """Train with RLPD.

    :param cfg: Hyper-parameters (see RLPDConfig).
    :param offline_buffer: Pre-populated buffer of expert transitions.
    :param run_dir: TensorBoard / checkpoint directory.
    :param stk_env: Pre-built STKVecEnv (shared with evaluation).
    :param make_evaluator: ``writer -> evaluator`` (e.g. a TrackSuiteEvaluator
        on ``stk_env.flat_env``); defaults to rl_mind's Evaluator.
    :returns: The Evaluator (holds the learning curve and the best checkpoint).
    """
    assert cfg.critic_subset_size <= cfg.n_critics, (
        f"critic_subset_size ({cfg.critic_subset_size}) must be ≤ n_critics ({cfg.n_critics})"
    )
    assert len(offline_buffer) >= int(cfg.batch_size * cfg.offline_fraction), (
        "offline_buffer must hold at least batch_size * offline_fraction transitions"
    )

    torch.manual_seed(cfg.seed)
    env = stk_env if stk_env is not None else VecEnv(cfg.env_name, cfg.n_envs, seed=cfg.seed)

    obs_dim = env.observation_dim
    act_dim = env.action_dim

    actor = SquashedGaussianActor(obs_dim, cfg.actor_hidden, act_dim)
    critics = EnsembleQNetwork(
        cfg.n_critics, obs_dim, cfg.critic_hidden, act_dim, layer_norm=cfg.use_layer_norm
    )
    targets = copy.deepcopy(critics).requires_grad_(False)

    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=cfg.lr)
    critic_params = list(critics.parameters())
    critic_optimizer = torch.optim.Adam(critic_params, lr=cfg.lr)

    auto_entropy = cfg.entropy_mode == "auto"
    log_alpha = torch.log(torch.ones(1) * cfg.init_entropy_coef).requires_grad_(auto_entropy)
    alpha_optimizer = torch.optim.Adam([log_alpha], lr=cfg.lr) if auto_entropy else None
    target_entropy = -float(act_dim)

    online_buffer = ReplayBuffer(cfg.buffer_size)
    collector = TransitionCollector(env, actor)

    n_offline = int(cfg.batch_size * cfg.offline_fraction)
    n_online = cfg.batch_size - n_offline
    # Critic updates per env step (actor/temperature: once per env step)
    utd = max(1, cfg.updates_per_round // cfg.steps_per_round)

    if run_dir is None:
        run_dir = f"runs/rlpd-{cfg.env_name}-S{cfg.seed}"
    writer = SummaryWriter(run_dir)
    if make_evaluator is not None:
        evaluator = make_evaluator(writer)
    else:
        eval_env = env if stk_env is not None else VecEnv(cfg.env_name, cfg.n_eval_envs, seed=cfg.seed + 100)
        evaluator = Evaluator(eval_env, every=cfg.eval_interval, run_dir=run_dir, writer=writer)
    # With a shared STK env, evaluation resets the env under the collector
    eval_shares_env = stk_env is not None
    last_log = 0

    pbar = tqdm(total=cfg.max_steps)
    while collector.steps < cfg.max_steps:
        online_buffer.add(collector.collect(cfg.steps_per_round))
        pbar.update(collector.steps - pbar.n)

        if len(online_buffer) < max(n_online, cfg.learning_starts):
            continue

        # UTD critic updates per env step, but a single actor / temperature
        # update per env step, on the last critic minibatch (as in the
        # official RLPD implementation: the actor gains little from a high
        # UTD, and it is half of the update cost).
        for i in range(cfg.updates_per_round):
            batch = Transitions.cat([
                offline_buffer.sample(n_offline),
                online_buffer.sample(n_online),
            ])
            alpha = log_alpha.exp().item()

            # --- critic update ---
            # Random subset Z of target critics for Bellman target (RED)
            z_idx = random.sample(range(cfg.n_critics), cfg.critic_subset_size)
            with torch.no_grad():
                next_act = actor(batch.next_obs)
                q_next = targets(batch.next_obs, next_act.value)[z_idx].min(dim=0).values
                # soft value: V(s') = min_Z Q'(s',ã') - α log π(ã'|s')
                # the (1-done) mask zeros the full V on terminal transitions
                v_next = q_next - alpha * next_act.log_prob
                target_q = cfg.reward_scale * batch.reward + cfg.gamma * (~batch.terminated).float() * v_next

            # MSE of each critic, summed over the ensemble
            q = critics(batch.obs, batch.action.value)
            critic_loss = (q - target_q.unsqueeze(0)).pow(2).mean(dim=1).sum()
            critic_optimizer.zero_grad()
            critic_loss.backward()
            nn.utils.clip_grad_norm_(critic_params, cfg.max_grad_norm)
            critic_optimizer.step()
            soft_update(critics, targets, cfg.tau)

            if (i + 1) % utd != 0:
                continue

            # --- actor update: mean Q over all E critics (paper line 23) ---
            action = actor(batch.obs)
            q_pi = critics(batch.obs, action.value).mean(dim=0)
            log_prob = action.log_prob
            actor_loss = (alpha * log_prob - q_pi).mean()
            actor_optimizer.zero_grad()
            actor_loss.backward()
            nn.utils.clip_grad_norm_(actor.parameters(), cfg.max_grad_norm)
            actor_optimizer.step()

            # --- entropy coefficient update ---
            if auto_entropy:
                alpha_loss = -(log_alpha * (log_prob.detach() + target_entropy)).mean()
                alpha_optimizer.zero_grad()
                alpha_loss.backward()
                alpha_optimizer.step()

        if collector.steps - last_log >= cfg.log_interval:
            last_log = collector.steps
            writer.add_scalar("loss/critic", critic_loss.item(), collector.steps)
            writer.add_scalar("loss/actor", actor_loss.item(), collector.steps)
            writer.add_scalar("entropy_coef", alpha, collector.steps)
            writer.add_scalar("buffer/online_size", len(online_buffer), collector.steps)
            # Diagnostics: Q on dataset actions vs policy actions (a large
            # gap = the actor exploits critic extrapolation), Bellman
            # targets, and what the deterministic policy does on each
            # action dimension.
            with torch.no_grad():
                writer.add_scalar("diag/q_data", q.mean().item(), collector.steps)
                writer.add_scalar("diag/q_policy", q_pi.mean().item(), collector.steps)
                writer.add_scalar("diag/target", target_q.mean().item(), collector.steps)
                writer.add_scalar("diag/reward_batch", batch.reward.mean().item(), collector.steps)
                det = actor.act(online_buffer.sample(n_online).obs)
                for j, name in enumerate(["accel", "steer"]):
                    writer.add_scalar(f"policy/{name}_mean", det[:, j].mean().item(), collector.steps)
                for j, name in enumerate(["brake", "drift", "nitro", "rescue"], start=2):
                    writer.add_scalar(f"policy/{name}_on", (det[:, j] > 0).float().mean().item(), collector.steps)
        if result := evaluator.run_if_needed(collector.steps, actor):
            if eval_shares_env:
                # Evaluation reset the env: the collector's current obs is
                # stale. Restart a fresh training episode instead of storing
                # a transition that jumps across the evaluation.
                collector._obs = None
            pbar.set_description(
                f"eval={result.mean:7.1f} best={evaluator.best_reward:7.1f}"
            )

    pbar.close()
    return evaluator
