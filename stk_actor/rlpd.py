"""RLPD: Reinforcement Learning with Prior Data (Ball et al., 2023).

Extends SAC with:
  - Two replay buffers: a frozen offline buffer (expert data) and a growing
    online buffer; each gradient step draws half the batch from each.
  - Layer norm in the critics, which stabilises Q-learning on mixed data.

The offline buffer is passed in already populated (see collect_expert_data()
in stk_actor/expert_collector.py, to be written separately).

Run from the repository root:

    PYTHONPATH=expert python -m stk_actor.rlpd
"""

import copy
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

from .sac import (
    SACConfig,
    SquashedGaussianActor,
    compute_actor_loss,
    compute_critic_loss,
)


# ---------------------------------------------------------------------------
# Critic with layer norm (RLPD paper §4)
# ---------------------------------------------------------------------------

class LayerNormQNetwork(nn.Module):
    """Q(s, a) with LayerNorm after each hidden activation.

    The RLPD paper shows that layer norm on the critics is critical when
    mixing offline expert data with fresh online transitions.
    """

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
    # Fraction of each batch drawn from the offline (expert) buffer.
    # 0.5 = strict 50 / 50 as in the original paper.
    offline_fraction: float = 0.5

    # RLPD can start training as soon as the online buffer has enough samples
    # for one half-batch.  Override the SAC default accordingly.
    learning_starts: int = 256


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def run_rlpd(
    cfg: RLPDConfig,
    offline_buffer: ReplayBuffer,
    run_dir: str | None = None,
    stk_env=None,       # optional pre-built STKVecEnv; if None, uses cfg.env_name
) -> Evaluator:
    """Train with RLPD.

    :param cfg: Hyper-parameters (see RLPDConfig).
    :param offline_buffer: Pre-populated buffer of expert transitions.
        Must already contain at least cfg.batch_size * cfg.offline_fraction
        transitions before this function is called.
    :param run_dir: TensorBoard / checkpoint directory.
    :returns: The Evaluator (holds the learning curve and the best checkpoint).
    """
    assert len(offline_buffer) >= int(cfg.batch_size * cfg.offline_fraction), (
        "offline_buffer must hold at least batch_size * offline_fraction transitions "
        "before calling run_rlpd()"
    )

    torch.manual_seed(cfg.seed)
    env = stk_env if stk_env is not None else VecEnv(cfg.env_name, cfg.n_envs, seed=cfg.seed)

    obs_dim = env.observation_dim
    act_dim = env.action_dim

    actor = SquashedGaussianActor(obs_dim, cfg.actor_hidden, act_dim)
    # RLPD uses layer-norm critics instead of plain MLPs
    critic_1 = LayerNormQNetwork(obs_dim, cfg.critic_hidden, act_dim)
    critic_2 = LayerNormQNetwork(obs_dim, cfg.critic_hidden, act_dim)
    target_1 = copy.deepcopy(critic_1)
    target_2 = copy.deepcopy(critic_2)

    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=cfg.lr)
    critic_optimizer = torch.optim.Adam(
        [*critic_1.parameters(), *critic_2.parameters()], lr=cfg.lr
    )

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
    evaluator = Evaluator(
        eval_env,
        every=cfg.eval_interval,
        run_dir=run_dir,
        writer=writer,
    )

    pbar = tqdm(total=cfg.max_steps)
    while collector.steps < cfg.max_steps:
        online_buffer.add(collector.collect(cfg.steps_per_round))
        pbar.update(collector.steps - pbar.n)

        # Wait until the online buffer has enough samples for one half-batch
        if len(online_buffer) < max(n_online, cfg.learning_starts):
            continue

        for _ in range(cfg.updates_per_round):
            # 50 / 50 mix: half offline (expert), half online (agent)
            batch = Transitions.cat([
                offline_buffer.sample(n_offline),
                online_buffer.sample(n_online),
            ])
            alpha = log_alpha.exp().item()

            # Critic update
            critic_loss = compute_critic_loss(
                cfg.gamma, alpha, batch, actor,
                (critic_1, critic_2), (target_1, target_2),
            )
            critic_optimizer.zero_grad()
            critic_loss.backward()
            nn.utils.clip_grad_norm_(
                [*critic_1.parameters(), *critic_2.parameters()], cfg.max_grad_norm
            )
            critic_optimizer.step()

            # Actor update
            actor_loss, log_prob = compute_actor_loss(
                alpha, batch.obs, actor, (critic_1, critic_2)
            )
            actor_optimizer.zero_grad()
            actor_loss.backward()
            nn.utils.clip_grad_norm_(actor.parameters(), cfg.max_grad_norm)
            actor_optimizer.step()

            # Entropy coefficient update
            if auto_entropy:
                alpha_loss = -(log_alpha * (log_prob.detach() + target_entropy)).mean()
                alpha_optimizer.zero_grad()
                alpha_loss.backward()
                alpha_optimizer.step()

            soft_update(critic_1, target_1, cfg.tau)
            soft_update(critic_2, target_2, cfg.tau)

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
