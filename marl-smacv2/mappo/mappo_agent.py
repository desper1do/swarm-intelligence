"""
MAPPO (Multi-Agent Proximal Policy Optimization)
Реализация для SMACv2 среды.

Особенности:
- On-policy алгоритм с централизованным критиком (CTDE)
- Рекуррентные сети (GRU) для актора и критика
- GAE (Generalized Advantage Estimation)
- Clipped objective для стабильности
- Parameter sharing между агентами
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from common.networks import RNNActor, RNNCritic, init_weights
from common.replay_buffer import RolloutBuffer


class MAPPOAgent:
    """
    MAPPO агент для SMACv2.
    Использует parameter sharing - все агенты используют общие веса.
    """

    def __init__(
        self,
        n_agents: int,
        obs_dim: int,
        n_actions: int,
        state_dim: int,
        hidden_dim: int = 64,
        rnn_hidden_dim: int = 64,
        lr_actor: float = 5e-4,
        lr_critic: float = 5e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: float = 0.2,
        value_clip: bool = True,
        value_clip_range: float = 0.2,
        entropy_coef: float = 0.01,
        value_loss_coef: float = 0.5,
        max_grad_norm: float = 10.0,
        device: str = "cuda",
    ):
        self.n_agents = n_agents
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.state_dim = state_dim
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_range = clip_range
        self.value_clip = value_clip
        self.value_clip_range = value_clip_range
        self.entropy_coef = entropy_coef
        self.value_loss_coef = value_loss_coef
        self.max_grad_norm = max_grad_norm
        self.device = device

        # AMP для ускорения на GPU
        self.use_amp = device == "cuda"
        self.scaler = torch.amp.GradScaler("cuda") if self.use_amp else None

        # Актор (parameter sharing)
        self.actor = RNNActor(obs_dim, n_actions, hidden_dim, rnn_hidden_dim).to(device)

        # Централизованный критик (видит полное состояние)
        self.critic = RNNCritic(state_dim, n_agents, hidden_dim, rnn_hidden_dim).to(device)

        # Оптимизаторы
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)

        # Инициализация
        self.actor.apply(init_weights)
        self.critic.apply(init_weights)

        self.train_step = 0

    def select_actions(
        self,
        observations: torch.Tensor,
        avail_actions: torch.Tensor,
        hidden_states: Optional[torch.Tensor] = None,
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, torch.Tensor, torch.Tensor]:
        """
        Выбирает действия для всех агентов.

        Args:
            observations: (n_agents, obs_dim)
            avail_actions: (n_agents, n_actions)
            hidden_states: (1, n_agents, rnn_hidden_dim)
            deterministic: детерминированный выбор

        Returns:
            actions: (n_agents,)
            log_probs: (n_agents,)
            new_hidden_states: (1, 1, rnn_hidden_dim)
        """
        # Добавляем batch dimension
        obs = observations.unsqueeze(0)  # (1, n_agents, obs_dim)

        if hidden_states is None:
            hidden_states = self.actor.init_hidden(obs.size(0)).to(self.device)

        # Получаем logits от актора
        logits, new_hidden_states = self.actor(obs, hidden_states)
        logits = logits.squeeze(0)  # (n_agents, n_actions)

        # Masking
        logits = logits.masked_fill(avail_actions == 0, -1e4)

        # Распределение
        dist = torch.distributions.Categorical(logits=logits)

        if deterministic:
            actions = dist.probs.argmax(dim=-1)
        else:
            actions = dist.sample()

        log_probs = dist.log_prob(actions)

        return actions.cpu().numpy(), log_probs.detach(), new_hidden_states.detach()

    def get_values(
        self,
        state: torch.Tensor,
        hidden_state: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Получает value-оценку состояния.

        Args:
            state: (state_dim,)
            hidden_state: (1, 1, rnn_hidden_dim)

        Returns:
            value: (1, 1)
            new_hidden_state: (1, 1, rnn_hidden_dim)
        """
        s = state.unsqueeze(0).unsqueeze(0)  # (1, 1, state_dim)

        if hidden_state is None:
            hidden_state = self.critic.init_hidden(1).to(self.device)

        value, new_hidden_state = self.critic(s, hidden_state)
        return value.view(1), new_hidden_state

    def evaluate_actions(
        self,
        observations: torch.Tensor,
        states: torch.Tensor,
        actions: torch.Tensor,
        avail_actions: torch.Tensor,
        actor_hidden: torch.Tensor,
        critic_hidden: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Оценивает действия для обучения.

        Args:
            observations: (batch, n_agents, obs_dim)
            states: (batch, state_dim)
            actions: (batch, n_agents)
            avail_actions: (batch, n_agents, n_actions)
            actor_hidden: (1, batch, rnn_hidden_dim)
            critic_hidden: (1, batch, rnn_hidden_dim)

        Returns:
            log_probs: (batch, n_agents)
            entropy: (batch,)
            values: (batch, 1)
        """
        # Актор
        logits, _ = self.actor(observations, actor_hidden)
        logits = logits.masked_fill(avail_actions == 0, -1e4)

        dist = torch.distributions.Categorical(logits=logits)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy().mean(dim=-1)

        # Критик
        values, _ = self.critic(states.unsqueeze(1), critic_hidden)
        values = values.squeeze(1)

        return log_probs, entropy, values

    def train(
        self,
        rollout_buffer: RolloutBuffer,
        n_epochs: int = 10,
        batch_size: int = 256,
    ) -> Dict[str, float]:
        """
        Обучает MAPPO используя данные из rollout buffer.

        Args:
            rollout_buffer: буфер с траекториями
            n_epochs: количество эпох обучения
            batch_size: размер батча

        Returns:
            losses: словарь с лоссами
        """
        # Получаем returns и advantages
        # Для этого нужно next_value
        # Упрощённая версия: считаем advantages внутри train

        # Squeeze n_envs=1 dimension
        squeeze = lambda x: x.squeeze(1) if x.ndim > 2 and x.shape[1] == 1 else x
        obs = squeeze(rollout_buffer.observations[:rollout_buffer.size])
        states = squeeze(rollout_buffer.states[:rollout_buffer.size])
        actions = squeeze(rollout_buffer.actions[:rollout_buffer.size])
        log_probs_old = squeeze(rollout_buffer.log_probs[:rollout_buffer.size])
        rewards = squeeze(rollout_buffer.rewards[:rollout_buffer.size])
        values_old = squeeze(rollout_buffer.values[:rollout_buffer.size])
        dones = squeeze(rollout_buffer.dones[:rollout_buffer.size])
        avail_actions = squeeze(rollout_buffer.avail_actions[:rollout_buffer.size])

        # Вычисляем advantages с GAE
        returns, advantages = self._compute_gae(rewards, values_old, dones)

        # Нормализация advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        total_actor_loss = 0
        total_critic_loss = 0
        total_entropy = 0
        n_updates = 0

        for epoch in range(n_epochs):
            # Создаём батчи
            total_size = rollout_buffer.size
            indices = np.arange(total_size)
            np.random.shuffle(indices)

            for start in range(0, total_size, batch_size):
                end = min(start + batch_size, total_size)
                batch_idx = indices[start:end]

                batch_obs = torch.FloatTensor(obs[batch_idx]).to(self.device)
                batch_states = torch.FloatTensor(states[batch_idx]).to(self.device)
                batch_actions = torch.LongTensor(actions[batch_idx]).to(self.device)
                batch_old_log_probs = torch.FloatTensor(log_probs_old[batch_idx]).to(self.device)
                batch_returns = torch.FloatTensor(returns[batch_idx]).to(self.device)
                batch_advantages = torch.FloatTensor(advantages[batch_idx]).to(self.device)
                batch_avail_actions = torch.FloatTensor(avail_actions[batch_idx]).to(self.device)
                batch_values_old = torch.FloatTensor(values_old[batch_idx]).to(self.device)

                # Инициализируем hidden states для батча
                actor_h = self.actor.init_hidden(batch_obs.size(0)).to(self.device)
                critic_h = self.critic.init_hidden(batch_states.size(0)).to(self.device)

                # Оцениваем действия (с AMP)
                with torch.amp.autocast("cuda", enabled=self.use_amp):
                    log_probs, entropy, values = self.evaluate_actions(
                        batch_obs,
                        batch_states,
                        batch_actions,
                        batch_avail_actions,
                        actor_h,
                        critic_h,
                    )

                    # PPO Loss
                    ratio = torch.exp(log_probs - batch_old_log_probs)
                    surr1 = ratio * batch_advantages.unsqueeze(-1)
                    surr2 = torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range) * batch_advantages.unsqueeze(-1)
                    actor_loss = -torch.min(surr1, surr2).mean()

                    # Value Loss
                    if self.value_clip:
                        values_clipped = batch_values_old + torch.clamp(
                            values - batch_values_old,
                            -self.value_clip_range,
                            self.value_clip_range,
                        )
                        value_loss1 = F.mse_loss(values, batch_returns, reduction="none")
                        value_loss2 = F.mse_loss(values_clipped, batch_returns, reduction="none")
                        critic_loss = torch.max(value_loss1, value_loss2).mean()
                    else:
                        critic_loss = F.mse_loss(values, batch_returns)

                    # Entropy bonus
                    entropy_loss = -entropy.mean()

                    # Total loss
                    loss = actor_loss + self.value_loss_coef * critic_loss + self.entropy_coef * entropy_loss

                # Optimization (с AMP)
                if self.use_amp:
                    self.actor_optimizer.zero_grad()
                    self.critic_optimizer.zero_grad()
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.actor_optimizer)
                    self.scaler.unscale_(self.critic_optimizer)
                    nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                    nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                    self.scaler.step(self.actor_optimizer)
                    self.scaler.step(self.critic_optimizer)
                    self.scaler.update()
                else:
                    self.actor_optimizer.zero_grad()
                    self.critic_optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                    nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                    self.actor_optimizer.step()
                    self.critic_optimizer.step()

                total_actor_loss += actor_loss.item()
                total_critic_loss += critic_loss.item()
                total_entropy += entropy.mean().item()
                n_updates += 1

        self.train_step += 1

        return {
            "actor_loss": total_actor_loss / max(n_updates, 1),
            "critic_loss": total_critic_loss / max(n_updates, 1),
            "entropy": total_entropy / max(n_updates, 1),
        }

    def _compute_gae(
        self,
        rewards: np.ndarray,
        values: np.ndarray,
        dones: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Вычисляет returns и advantages с помощью GAE.

        Args:
            rewards: (size, 1)
            values: (size, 1)
            dones: (size, 1)

        Returns:
            returns: (size, 1)
            advantages: (size, 1)
        """
        returns = np.zeros_like(rewards)
        advantages = np.zeros_like(rewards)

        last_gae = 0
        next_value = 0  # Terminal state

        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_val = next_value
            else:
                next_val = values[t + 1]

            delta = rewards[t] + self.gamma * next_val * (1 - dones[t]) - values[t]
            last_gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * last_gae
            advantages[t] = last_gae
            returns[t] = advantages[t] + values[t]

        return returns, advantages

    def save(self, path: str):
        """Сохраняет модель."""
        torch.save({
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "train_step": self.train_step,
        }, path)

    def load(self, path: str):
        """Загружает модель."""
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint["actor"])
        self.critic.load_state_dict(checkpoint["critic"])
        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
        self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])
        self.train_step = checkpoint["train_step"]
