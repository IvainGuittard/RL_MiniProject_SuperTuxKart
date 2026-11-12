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

    PYTHONPATH=. python -m stk_actor.learn rlpd --expert team3 --offline-steps 50000
"""

import copy
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

from .sac import SACConfig, SquashedGaussianActor, ContinuousQNetwork


# ---------------------------------------------------------------------------
# Critic with LayerNorm (RLPD paper §4)
# ---------------------------------------------------------------------------

class LayerNormQNetwork(nn.Module):
    """Q(s, a) with LayerNorm after each hidden activation."""

    def __init__(self, obs_dim: int, hidden: tuple[int, ...], action_dim: int):
        super().__init__()
        dims = [obs_dim + action_dim, *hidden]
        layers: list[nn.Module] = []
        for d_in, d_out in zip(dims[:-1], dims[1:]):
            layers += [nn.Linear(d_in, d_out), nn.LayerNorm(d_out), nn.ReLU()]
        layers.append(nn.Linear(hidden[-1], 1))
        self.model = nn.Sequential(*layers)

    def forward(self, obs: Tensor, action: Tensor) -> Tensor:
        return self.model(torch.cat([obs, action], dim=1)).squeeze(-1)


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

    # --- critic ensemble (Random Ensemble Distillation) ---
    # E in the paper: number of critics to train.
    n_critics: int = 2
    # Z in the paper: size of the random subset used to compute the target.
    # Must be ≤ n_critics.  Z=2 with E=2 is standard twin-critic SAC.
    critic_subset_size: int = 2

    # --- ablation flags ---
    use_layer_norm: bool = True   # LayerNorm in critics (key RLPD contribution)
    # entropy_mode inherited from SACConfig: "auto" or "fixed"


# ---------------------------------------------------------------------------
# Ensemble helpers
# ---------------------------------------------------------------------------

def _make_critic(cfg: RLPDConfig, obs_dim: int, act_dim: int) -> nn.Module:
    if cfg.use_layer_norm:
        return LayerNormQNetwork(obs_dim, cfg.critic_hidden, act_dim)
    return ContinuousQNetwork(obs_dim, cfg.critic_hidden, act_dim)


def _critic_target(
    targets: list[nn.Module],
    subset_indices: list[int],
    next_obs: Tensor,
    next_action: Tensor,
) -> Tensor:
    """Min Q over a random subset of target critics."""
    qs = torch.stack([targets[i](next_obs, next_action) for i in subset_indices])
    return qs.min(dim=0).values


def _critic_loss(
    critics: list[nn.Module],
    obs: Tensor,
    action: Tensor,
    target_q: Tensor,
) -> Tensor:
    """MSE loss summed over all E critics."""
    return sum(
        F.mse_loss(c(obs, action), target_q) for c in critics
    )


def _actor_loss(
    critics: list[nn.Module],
    alpha: float,
    obs: Tensor,
    actor: SquashedGaussianActor,
) -> tuple[Tensor, Tensor]:
    """Actor loss using the mean Q over all E critics (paper line 23)."""
    action = actor(obs)
    q_values = torch.stack([c(obs, action.value) for c in critics])
    q_mean = q_values.mean(dim=0)
    loss = (alpha * action.log_prob - q_mean).mean()
    return loss, action.log_prob


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def run_rlpd(
    cfg: RLPDConfig,
    offline_buffer: ReplayBuffer,
    run_dir: str | None = None,
    stk_env=None,
) -> Evaluator:
    """Train with RLPD.

    :param cfg: Hyper-parameters (see RLPDConfig).
    :param offline_buffer: Pre-populated buffer of expert transitions.
    :param run_dir: TensorBoard / checkpoint directory.
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
    critics = [_make_critic(cfg, obs_dim, act_dim) for _ in range(cfg.n_critics)]
    targets = [copy.deepcopy(c) for c in critics]
    for t in targets:
        t.requires_grad_(False)

    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=cfg.lr)
    critic_params = [p for c in critics for p in c.parameters()]
    critic_optimizer = torch.optim.Adam(critic_params, lr=cfg.lr)

    auto_entropy = cfg.entropy_mode == "auto"
    log_alpha = torch.log(torch.ones(1) * cfg.init_entropy_coef).requires_grad_(auto_entropy)
    alpha_optimizer = torch.optim.Adam([log_alpha], lr=cfg.lr) if auto_entropy else None
    target_entropy = -float(act_dim)

    online_buffer = ReplayBuffer(cfg.buffer_size)
    collector = TransitionCollector(env, actor)

    n_offline = int(cfg.batch_size * cfg.offline_fraction)
    n_online = cfg.batch_size - n_offline

    if run_dir is None:
        run_dir = f"runs/rlpd-{cfg.env_name}-S{cfg.seed}"
    writer = SummaryWriter(run_dir)
    eval_env = env if stk_env is not None else VecEnv(cfg.env_name, cfg.n_eval_envs, seed=cfg.seed + 100)
    evaluator = Evaluator(eval_env, every=cfg.eval_interval, run_dir=run_dir, writer=writer)

    pbar = tqdm(total=cfg.max_steps)
    while collector.steps < cfg.max_steps:
        online_buffer.add(collector.collect(cfg.steps_per_round))
        pbar.update(collector.steps - pbar.n)

        if len(online_buffer) < max(n_online, cfg.learning_starts):
            continue

        for _ in range(cfg.updates_per_round):
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
                q_next = _critic_target(targets, z_idx, batch.next_obs, next_act.value)
                # soft value: V(s') = min_Z Q'(s',ã') - α log π(ã'|s')
                # the (1-done) mask zeros the full V on terminal transitions
                v_next = q_next - alpha * next_act.log_prob
                target_q = batch.reward + cfg.gamma * (~batch.terminated).float() * v_next

            critic_loss = _critic_loss(critics, batch.obs, batch.action.value, target_q)
            critic_optimizer.zero_grad()
            critic_loss.backward()
            nn.utils.clip_grad_norm_(critic_params, cfg.max_grad_norm)
            critic_optimizer.step()

            # --- actor update (mean over all E critics) ---
            actor_loss, log_prob = _actor_loss(critics, alpha, batch.obs, actor)
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

            for c, t in zip(critics, targets):
                soft_update(c, t, cfg.tau)

        writer.add_scalar("loss/critic", critic_loss.item(), collector.steps)
        writer.add_scalar("loss/actor", actor_loss.item(), collector.steps)
        writer.add_scalar("entropy_coef", alpha, collector.steps)
        writer.add_scalar("buffer/online_size", len(online_buffer), collector.steps)
        if result := evaluator.run_if_needed(collector.steps, actor):
            pbar.set_description(
                f"eval={result.mean:7.1f} best={evaluator.best_reward:7.1f}"
            )

    pbar.close()
    return evaluator
