"""SAC implementation. Run via learn.py:

    PYTHONPATH=. python -m stk_actor.learn sac
"""

import copy
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.distributions import Independent, Normal, TanhTransform, TransformedDistribution
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

from rl_mind.core import Actor, StochasticAction
from rl_mind.nn import build_mlp, soft_update
from rl_mind.env import VecEnv
from rl_mind.data import ReplayBuffer, Transitions
from rl_mind.collectors import TransitionCollector
from rl_mind.evaluation import Evaluator


# ---------------------------------------------------------------------------
# Actor
# ---------------------------------------------------------------------------

class SquashedGaussianActor(Actor[StochasticAction]):
    """Squashed-Gaussian stochastic actor for SAC."""

    def __init__(
        self,
        obs_dim: int,
        hidden: tuple[int, ...],
        action_dim: int,
        min_std: float = 1e-4,
    ):
        super().__init__()
        self.backbone = build_mlp([obs_dim, *hidden], output_activation=nn.ReLU())
        self.mean_layer = nn.Linear(hidden[-1], action_dim)
        self.std_layer = nn.Linear(hidden[-1], action_dim)
        self.softplus = nn.Softplus()
        self.min_std = min_std
        self.tanh = TanhTransform(cache_size=1)

    def dist(self, obs: Tensor) -> TransformedDistribution:
        features = self.backbone(obs)
        mean = self.mean_layer(features)
        std = self.softplus(self.std_layer(features)) + self.min_std
        normal = Independent(Normal(mean, std), 1)
        return TransformedDistribution(normal, [self.tanh])

    def forward(self, obs: Tensor) -> StochasticAction:
        dist = self.dist(obs)
        action = dist.rsample()
        log_prob = dist.log_prob(action)
        self.tanh._cached_x_y = [None, None]
        return StochasticAction(value=action, log_prob=log_prob)

    def act(self, obs: Tensor) -> Tensor:
        features = self.backbone(obs)
        return torch.tanh(self.mean_layer(features))


# ---------------------------------------------------------------------------
# Critic
# ---------------------------------------------------------------------------

class ContinuousQNetwork(nn.Module):
    """Twin-critic Q(s, a) for continuous actions."""

    def __init__(self, obs_dim: int, hidden: tuple[int, ...], action_dim: int):
        super().__init__()
        self.model = build_mlp([obs_dim + action_dim, *hidden, 1])

    def forward(self, obs: Tensor, action: Tensor) -> Tensor:
        return self.model(torch.cat([obs, action], dim=1)).squeeze(-1)


# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------

def compute_critic_loss(
    gamma: float,
    alpha: float,
    batch: Transitions,
    actor: SquashedGaussianActor,
    critics: tuple[ContinuousQNetwork, ContinuousQNetwork],
    target_critics: tuple[ContinuousQNetwork, ContinuousQNetwork],
) -> Tensor:
    q1, q2 = critics
    tq1, tq2 = target_critics

    with torch.no_grad():
        next_action = actor(batch.next_obs)
        q_next = torch.min(
            tq1(batch.next_obs, next_action.value),
            tq2(batch.next_obs, next_action.value),
        )
        v_next = q_next - alpha * next_action.log_prob
        target = batch.reward + gamma * (~batch.terminated).float() * v_next

    loss_q1 = F.mse_loss(q1(batch.obs, batch.action.value), target)
    loss_q2 = F.mse_loss(q2(batch.obs, batch.action.value), target)
    return loss_q1 + loss_q2


def compute_actor_loss(
    alpha: float,
    obs: Tensor,
    actor: SquashedGaussianActor,
    critics: tuple[ContinuousQNetwork, ContinuousQNetwork],
) -> tuple[Tensor, Tensor]:
    q1, q2 = critics
    action = actor(obs)
    q = torch.min(q1(obs, action.value), q2(obs, action.value))
    actor_loss = (alpha * action.log_prob - q).mean()
    return actor_loss, action.log_prob


# ---------------------------------------------------------------------------
# Config & training loop
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SACConfig:
    env_name: str = "stk"
    seed: int = 1

    max_steps: int = 300_000
    n_envs: int = 8
    steps_per_round: int = 32
    updates_per_round: int = 32
    learning_starts: int = 10_000

    buffer_size: int = 1_000_000
    batch_size: int = 256

    gamma: float = 0.99
    tau: float = 0.005
    max_grad_norm: float = 0.5

    entropy_mode: str = "auto"
    init_entropy_coef: float = 0.2

    actor_hidden: tuple[int, ...] = (256, 256)
    critic_hidden: tuple[int, ...] = (256, 256)
    lr: float = 3e-4

    eval_interval: int = 5_000
    n_eval_envs: int = 10


def run_sac(
    cfg: SACConfig,
    run_dir: str | None = None,
    stk_env=None,       # optional pre-built STKVecEnv; if None, uses cfg.env_name
    make_evaluator=None,  # optional ``writer -> evaluator`` (e.g. TrackSuiteEvaluator)
):
    torch.manual_seed(cfg.seed)
    env = stk_env if stk_env is not None else VecEnv(cfg.env_name, cfg.n_envs, seed=cfg.seed)

    obs_dim = env.observation_dim
    act_dim = env.action_dim

    actor = SquashedGaussianActor(obs_dim, cfg.actor_hidden, act_dim)
    critic_1 = ContinuousQNetwork(obs_dim, cfg.critic_hidden, act_dim)
    critic_2 = ContinuousQNetwork(obs_dim, cfg.critic_hidden, act_dim)
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

    collector = TransitionCollector(env, actor)
    buffer = ReplayBuffer(cfg.buffer_size)

    if run_dir is None:
        run_dir = f"runs/sac-{cfg.env_name}-S{cfg.seed}"
    writer = SummaryWriter(run_dir)
    # With a custom stk_env we reuse the training env for evaluation (pystk2
    # is a singleton; we cannot create a second independent instance).
    if make_evaluator is not None:
        evaluator = make_evaluator(writer)
    else:
        eval_env = env if stk_env is not None else VecEnv(cfg.env_name, cfg.n_eval_envs, seed=cfg.seed + 100)
        evaluator = Evaluator(
            eval_env,
            every=cfg.eval_interval,
            run_dir=run_dir,
            writer=writer,
        )
    eval_shares_env = stk_env is not None

    pbar = tqdm(total=cfg.max_steps)
    while collector.steps < cfg.max_steps:
        buffer.add(collector.collect(cfg.steps_per_round))
        pbar.update(collector.steps - pbar.n)
        if len(buffer) < cfg.learning_starts:
            continue

        for _ in range(cfg.updates_per_round):
            batch = buffer.sample(cfg.batch_size)
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
            actor_loss, log_prob = compute_actor_loss(alpha, batch.obs, actor, (critic_1, critic_2))
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
        if result := evaluator.run_if_needed(collector.steps, actor):
            if eval_shares_env:
                # Evaluation reset the shared env: restart a training episode
                # rather than store a transition spanning the evaluation.
                collector._obs = None
            pbar.set_description(
                f"eval={result.mean:7.1f} best={evaluator.best_reward:7.1f}"
            )

    pbar.close()
    return evaluator


