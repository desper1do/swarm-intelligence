"""
MADDPG (Multi-Agent Deep Deterministic Policy Gradient)
Реализация для SMACv2 среды.

Особенности:
- Централизованное обучение, децентрализованное выполнение (CTDE)
- Каждый агент имеет свой актор и критик
- Критик получает наблюдения и действия ВСЕХ агентов
- Актор получает только локальное наблюдение
- Поддержка action masking для недоступных действий
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

from common.networks import MLPActor, MLPCritic, init_weights
from common.replay_buffer import ReplayBuffer


class MADDPGAgent:
    """
    MADDPG агент для кооперативного сценария SMACv2.
    """

    def __init__(
        self,
        agent_id: int,
        obs_dim: int,
        action_dim: int,
        total_obs_dim: int,
        total_action_dim: int,
        hidden_dims: List[int] = [256, 128],
        lr_actor: float = 1e-4,
        lr_critic: float = 1e-3,
        gamma: float = 0.99,
        tau: float = 0.005,
        device: str = "cuda",
    ):
        self.agent_id = agent_id
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.device = device

        # Актор (децентрализованный - только локальное наблюдение)
        self.actor = MLPActor(obs_dim, action_dim, hidden_dims, use_layer_norm=True).to(device)
        self.actor_target = MLPActor(obs_dim, action_dim, hidden_dims, use_layer_norm=True).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)

        # Критик (централизованный - все наблюдения и действия)
        self.critic = MLPCritic(total_obs_dim, total_action_dim, hidden_dims, use_layer_norm=True).to(device)
        self.critic_target = MLPCritic(total_obs_dim, total_action_dim, hidden_dims, use_layer_norm=True).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)

        # Инициализация весов
        self.actor.apply(init_weights)
        self.critic.apply(init_weights)

    def select_action(
        self,
        obs: torch.Tensor,
        avail_actions: torch.Tensor,
        explore: bool = True,
        noise_scale: float = 0.1,
    ) -> torch.Tensor:
        """
        Выбирает действие с исследованием.

        Args:
            obs: (obs_dim,) - наблюдение агента
            avail_actions: (action_dim,) - маска доступных действий
            explore: включить исследование
            noise_scale: масштаб шума Ornstein-Uhlenbeck

        Returns:
            action: (action_dim,) - действие в диапазоне [-1, 1]
        """
        with torch.no_grad():
            action = self.actor(obs)

        if explore:
            noise = torch.randn_like(action) * noise_scale
            action = torch.clamp(action + noise, -1, 1)

        # Masking: обнуляем недоступные действия
        action = action * avail_actions

        return action

    def soft_update(self):
        """Мягкое обновление target сетей."""
        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)


class MADDPGTrainer:
    """
    Тренер MADDPG для обучения команды агентов в SMACv2.
    """

    def __init__(
        self,
        n_agents: int,
        obs_dim: int,
        n_actions: int,
        state_dim: int,
        hidden_dims: List[int] = [256, 128],
        buffer_capacity: int = 1000000,
        batch_size: int = 256,
        lr_actor: float = 1e-4,
        lr_critic: float = 1e-3,
        gamma: float = 0.99,
        tau: float = 0.005,
        noise_scale: float = 0.5,
        noise_decay: float = 0.999,
        warmup_steps: int = 2000,
        device: str = "cuda",
    ):
        self.n_agents = n_agents
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.state_dim = state_dim
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.noise_scale = noise_scale
        self.noise_decay = noise_decay
        self.warmup_steps = warmup_steps
        self.device = device

        # AMP для ускорения на GPU
        self.use_amp = device == "cuda"
        self.scaler = torch.amp.GradScaler("cuda") if self.use_amp else None

        # Размеры для централизованного критика
        # Критик видит obs ВСЕХ агентов и действия ВСЕХ агентов (one-hot)
        total_obs_dim = n_agents * obs_dim
        total_action_dim = n_agents * n_actions

        # Создаём агентов
        self.agents: List[MADDPGAgent] = []
        for i in range(n_agents):
            agent = MADDPGAgent(
                agent_id=i,
                obs_dim=obs_dim,
                action_dim=n_actions,  # Выходим в пространство дискретных действий
                total_obs_dim=total_obs_dim,
                total_action_dim=total_action_dim,
                hidden_dims=hidden_dims,
                lr_actor=lr_actor,
                lr_critic=lr_critic,
                gamma=gamma,
                tau=tau,
                device=device,
            )
            self.agents.append(agent)

        # Replay Buffer
        self.replay_buffer = ReplayBuffer(
            capacity=buffer_capacity,
            n_agents=n_agents,
            obs_dim=obs_dim,
            action_dim=n_actions,
            state_dim=state_dim,
            device=device,
        )

        self.episode_count = 0
        self.train_step = 0

    def select_actions(
        self,
        observations: np.ndarray,
        avail_actions: np.ndarray,
        explore: bool = True,
    ) -> np.ndarray:
        """
        Выбирает действия для всех агентов.

        Args:
            observations: (n_agents, obs_dim)
            avail_actions: (n_agents, n_actions) - маски
            explore: исследование

        Returns:
            actions: (n_agents,) - дискретные действия
        """
        obs_tensor = torch.FloatTensor(observations).to(self.device)
        avail_tensor = torch.FloatTensor(avail_actions).to(self.device)

        # Получаем continuous actions от каждого агента
        # Во время warmup используем полностью случайные действия (максимальный exploration)
        is_warmup = self.train_step < self.warmup_steps
        effective_noise = 1.0 if (is_warmup and explore) else self.noise_scale

        continuous_actions = []
        for i, agent in enumerate(self.agents):
            action = agent.select_action(
                obs_tensor[i],
                avail_tensor[i],
                explore=explore,
                noise_scale=effective_noise,
            )
            continuous_actions.append(action.cpu().numpy())

        continuous_actions = np.array(continuous_actions)

        # Преобразуем continuous -> discrete через argmax с учётом маски
        # Устанавливаем недоступные действия в очень маленькое значение
        masked_actions = continuous_actions.copy()
        masked_actions[avail_actions == 0] = -1e10

        discrete_actions = np.argmax(masked_actions, axis=1)

        return discrete_actions

    def store_transition(
        self,
        state: np.ndarray,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        next_obs: np.ndarray,
        done: bool,
    ):
        """Сохраняет переход в буфер."""
        # Преобразуем дискретные действия в one-hot для буфера
        action_onehot = np.zeros((self.n_agents, self.n_actions), dtype=np.float32)
        for i, a in enumerate(action):
            action_onehot[i, a] = 1.0

        self.replay_buffer.add(state, obs, action_onehot, reward, next_state, next_obs, done)

    def train(self) -> Dict[str, float]:
        """
        Выполняет одно обновление всех агентов.

        Returns:
            losses: словарь с лоссами
        """
        if len(self.replay_buffer) < self.batch_size:
            return {}

        batch = self.replay_buffer.sample(self.batch_size)

        actor_losses = []
        critic_losses = []

        for agent_id, agent in enumerate(self.agents):
            # ===== Обновление критика =====
            # Во время warmup обучаем только критика (больший шум)
            is_warmup = self.train_step < self.warmup_steps

            with torch.no_grad(), torch.amp.autocast("cuda", enabled=self.use_amp):
                # Целевые действия от target акторов
                next_actions = []
                for i, a in enumerate(self.agents):
                    next_obs_i = batch["next_observations"][:, i, :]
                    next_act = a.actor_target(next_obs_i)
                    next_actions.append(next_act)
                next_actions = torch.stack(next_actions, dim=1)  # (batch, n_agents, n_actions)

                # Target Q-значение
                next_q = agent.critic_target(
                    batch["next_observations"].view(self.batch_size, -1),
                    next_actions.view(self.batch_size, -1),
                )
                target_q = batch["rewards"] + self.gamma * batch["dones"] * next_q

            # Текущее Q-значение (с AMP)
            with torch.amp.autocast("cuda", enabled=self.use_amp):
                current_q = agent.critic(
                    batch["observations"].view(self.batch_size, -1),
                    batch["actions"].view(self.batch_size, -1),
                )
                critic_loss = F.mse_loss(current_q, target_q)

            # AMP backward для критика
            if self.use_amp:
                agent.critic_optimizer.zero_grad()
                self.scaler.scale(critic_loss).backward()
                self.scaler.unscale_(agent.critic_optimizer)
                torch.nn.utils.clip_grad_norm_(agent.critic.parameters(), 0.5)
                self.scaler.step(agent.critic_optimizer)
            else:
                agent.critic_optimizer.zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(agent.critic.parameters(), 0.5)
                agent.critic_optimizer.step()

            critic_losses.append(critic_loss.item())

            # ===== Обновление актора (пропускаем во время warmup) =====
            if not is_warmup:
                with torch.amp.autocast("cuda", enabled=self.use_amp):
                    actions_for_critic = batch["actions"].clone()
                    agent_action = agent.actor(batch["observations"][:, agent_id, :])
                    actions_for_critic[:, agent_id, :] = agent_action

                    actor_loss = -agent.critic(
                        batch["observations"].view(self.batch_size, -1),
                        actions_for_critic.view(self.batch_size, -1),
                    ).mean()

                # AMP backward для актора
                if self.use_amp:
                    agent.actor_optimizer.zero_grad()
                    self.scaler.scale(actor_loss).backward()
                    self.scaler.unscale_(agent.actor_optimizer)
                    torch.nn.utils.clip_grad_norm_(agent.actor.parameters(), 0.5)
                    self.scaler.step(agent.actor_optimizer)
                else:
                    agent.actor_optimizer.zero_grad()
                    actor_loss.backward()
                    torch.nn.utils.clip_grad_norm_(agent.actor.parameters(), 0.5)
                    agent.actor_optimizer.step()

                actor_losses.append(actor_loss.item())

        # AMP scaler update
        if self.use_amp:
            self.scaler.update()

        # Soft update target сетей
        for agent in self.agents:
            agent.soft_update()

        # Decay noise (медленнее)
        self.noise_scale *= self.noise_decay
        self.noise_scale = max(self.noise_scale, 0.05)  # Минимальный шум не 0

        self.train_step += 1

        return {
            "actor_loss": np.mean(actor_losses),
            "critic_loss": np.mean(critic_losses),
            "noise_scale": self.noise_scale,
        }

    def save(self, path: str):
        """Сохраняет модели всех агентов."""
        save_dict = {
            "agents": [{
                "actor": agent.actor.state_dict(),
                "critic": agent.critic.state_dict(),
                "actor_optimizer": agent.actor_optimizer.state_dict(),
                "critic_optimizer": agent.critic_optimizer.state_dict(),
            } for agent in self.agents],
            "episode_count": self.episode_count,
            "train_step": self.train_step,
            "noise_scale": self.noise_scale,
        }
        torch.save(save_dict, path)

    def load(self, path: str):
        """Загружает модели всех агентов."""
        checkpoint = torch.load(path, map_location=self.device)
        for i, agent_state in enumerate(checkpoint["agents"]):
            self.agents[i].actor.load_state_dict(agent_state["actor"])
            self.agents[i].critic.load_state_dict(agent_state["critic"])
            self.agents[i].actor_optimizer.load_state_dict(agent_state["actor_optimizer"])
            self.agents[i].critic_optimizer.load_state_dict(agent_state["critic_optimizer"])
            # Обновляем target сети
            self.agents[i].actor_target.load_state_dict(agent_state["actor"])
            self.agents[i].critic_target.load_state_dict(agent_state["critic"])

        self.episode_count = checkpoint["episode_count"]
        self.train_step = checkpoint["train_step"]
        self.noise_scale = checkpoint["noise_scale"]
