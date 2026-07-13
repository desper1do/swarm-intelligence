"""
QMIX (Q-value MIXing)
Реализация для SMACv2 среды.

Особенности:
- DRQN (Deep Recurrent Q-Network) для каждого агента
- Mixing network для комбинирования Q-значений агентов
- Централизованное обучение с децентрализованным выполнением
- Поддержка action masking и двойного DQN
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

from common.networks import DRQNNetwork, MixerNetwork, init_weights
from common.replay_buffer import EpisodeBuffer


class QMIXAgent:
    """
    QMIX агент с DRQN для SMACv2.
    """

    def __init__(
        self,
        n_agents: int,
        obs_dim: int,
        n_actions: int,
        state_dim: int,
        hidden_dim: int = 64,
        rnn_hidden_dim: int = 64,
        mixer_hidden_dim: int = 32,
        hypernet_hidden_dim: int = 64,
        gamma: float = 0.99,
        lr: float = 5e-4,
        tau: float = 0.005,
        device: str = "cuda",
    ):
        self.n_agents = n_agents
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.state_dim = state_dim
        self.gamma = gamma
        self.tau = tau
        self.device = device

        # AMP для ускорения на GPU
        self.use_amp = device == "cuda"
        self.scaler = torch.amp.GradScaler("cuda") if self.use_amp else None

        # Q-сети для каждого агента (делим веса между агентами - параметр sharing)
        self.q_network = DRQNNetwork(obs_dim, n_actions, hidden_dim, rnn_hidden_dim).to(device)
        self.q_target = DRQNNetwork(obs_dim, n_actions, hidden_dim, rnn_hidden_dim).to(device)
        self.q_target.load_state_dict(self.q_network.state_dict())

        # Mixer network
        self.mixer = MixerNetwork(n_agents, state_dim, mixer_hidden_dim, hypernet_hidden_dim).to(device)
        self.mixer_target = MixerNetwork(n_agents, state_dim, mixer_hidden_dim, hypernet_hidden_dim).to(device)
        self.mixer_target.load_state_dict(self.mixer.state_dict())

        # Оптимизатор
        self.optimizer = optim.Adam(
            list(self.q_network.parameters()) + list(self.mixer.parameters()),
            lr=lr,
        )

        # Инициализация
        self.q_network.apply(init_weights)
        self.mixer.apply(init_weights)

        self.train_step = 0

    def select_actions(
        self,
        observations: torch.Tensor,
        avail_actions: torch.Tensor,
        hidden_states: Optional[torch.Tensor] = None,
        epsilon: float = 0.0,
    ) -> Tuple[np.ndarray, torch.Tensor]:
        """
        Выбирает действия epsilon-greedy.

        Args:
            observations: (n_agents, obs_dim)
            avail_actions: (n_agents, n_actions)
            hidden_states: (1, n_agents, rnn_hidden_dim)
            epsilon: параметр исследования

        Returns:
            actions: (n_agents,) - дискретные действия
            new_hidden_states: (1, n_agents, rnn_hidden_dim)
        """
        with torch.no_grad():
            q_values, new_hidden_states = self.q_network(observations, hidden_states)

        # Masking
        q_values = q_values.masked_fill(avail_actions == 0, -1e4)

        if np.random.random() < epsilon:
            # Случайное действие из доступных
            actions = []
            for i in range(self.n_agents):
                avail = avail_actions[i].cpu().numpy()
                avail_indices = np.where(avail > 0)[0]
                if len(avail_indices) > 0:
                    actions.append(np.random.choice(avail_indices))
                else:
                    actions.append(0)
            actions = np.array(actions)
        else:
            actions = q_values.argmax(dim=-1).cpu().numpy()

        return actions, new_hidden_states

    def train(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Обучает QMIX на батче эпизодов.

        Args:
            batch: словарь с эпизодами
                - obs: (batch, max_seq, n_agents, obs_dim)
                - states: (batch, max_seq, state_dim)
                - actions: (batch, max_seq, n_agents)
                - rewards: (batch, max_seq, 1)
                - next_obs: (batch, max_seq, n_agents, obs_dim)
                - next_states: (batch, max_seq, state_dim)
                - dones: (batch, max_seq, 1)
                - avail_actions: (batch, max_seq, n_agents, n_actions)
                - next_avail_actions: (batch, max_seq, n_agents, n_actions)
                - mask: (batch, max_seq, 1)

        Returns:
            losses: словарь с лоссами
        """
        batch_size = batch["obs"].size(0)
        max_seq = batch["obs"].size(1)

        # === Вычисляем Q_tot (с AMP) ===
        with torch.amp.autocast("cuda", enabled=self.use_amp):
            # Q-значения для всех агентов
            q_values_list = []
            for i in range(self.n_agents):
                obs_i = batch["obs"][:, :, i, :]  # (batch, max_seq, obs_dim)
                q_i, _ = self.q_network(obs_i)
                q_values_list.append(q_i)

            # (batch, max_seq, n_agents, n_actions)
            q_values = torch.stack(q_values_list, dim=2)

            # Выбираем Q для выбранных действий
            actions = batch["actions"].unsqueeze(-1)  # (batch, max_seq, n_agents, 1)
            chosen_q = q_values.gather(-1, actions).squeeze(-1)  # (batch, max_seq, n_agents)

            # Mixer: flatten seq dimension
            states = batch["states"]  # (batch, max_seq, state_dim)
            chosen_q_flat = chosen_q.view(batch_size * max_seq, self.n_agents)  # (batch*seq, n_agents)
            states_flat = states.view(batch_size * max_seq, self.state_dim)  # (batch*seq, state_dim)
            q_tot = self.mixer(chosen_q_flat, states_flat)
            q_tot = q_tot.view(batch_size, max_seq, 1)

        # === Вычисляем target Q_tot ===
        with torch.no_grad(), torch.amp.autocast("cuda", enabled=self.use_amp):
            # Target Q-значения
            target_q_list = []
            for i in range(self.n_agents):
                next_obs_i = batch["next_obs"][:, :, i, :]
                q_i, _ = self.q_target(next_obs_i)
                target_q_list.append(q_i)

            target_q_values = torch.stack(target_q_list, dim=2)

            # Masking для следующих действий
            target_q_values = target_q_values.masked_fill(batch["next_avail_actions"] == 0, -1e4)

            # Двойной DQN: используем online сеть для выбора действий
            online_q_list = []
            for i in range(self.n_agents):
                next_obs_i = batch["next_obs"][:, :, i, :]
                q_i, _ = self.q_network(next_obs_i)
                online_q_list.append(q_i)
            online_q_values = torch.stack(online_q_list, dim=2)
            online_q_values = online_q_values.masked_fill(batch["next_avail_actions"] == 0, -1e4)

            # Выбираем действия по online сети
            next_actions = online_q_values.argmax(dim=-1, keepdim=True)
            target_q_chosen = target_q_values.gather(-1, next_actions).squeeze(-1)

            # Target mixer: flatten seq dimension
            next_states = batch["next_states"]
            target_q_chosen_flat = target_q_chosen.view(batch_size * max_seq, self.n_agents)
            target_q_tot = self.mixer_target(
                target_q_chosen_flat,
                next_states.view(batch_size * max_seq, self.state_dim),
            )
            target_q_tot = target_q_tot.view(batch_size, max_seq, 1)

            # TD target
            rewards = batch["rewards"]
            dones = batch["dones"]
            td_target = rewards + self.gamma * (1 - dones) * target_q_tot

        # === Loss ===
        with torch.amp.autocast("cuda", enabled=self.use_amp):
            mask = batch["mask"]
            loss = (F.mse_loss(q_tot, td_target, reduction="none") * mask).sum() / mask.sum()

        # === Optimization (с AMP) ===
        if self.use_amp:
            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                list(self.q_network.parameters()) + list(self.mixer.parameters()),
                10.0,
            )
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(self.q_network.parameters()) + list(self.mixer.parameters()),
                10.0,
            )
            self.optimizer.step()

        # Soft update
        self._soft_update()

        self.train_step += 1

        return {
            "qmix_loss": loss.item(),
            "q_tot_mean": q_tot.mean().item(),
            "td_target_mean": td_target.mean().item(),
        }

    def _soft_update(self):
        """Мягкое обновление target сетей."""
        for param, target_param in zip(self.q_network.parameters(), self.q_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for param, target_param in zip(self.mixer.parameters(), self.mixer_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def save(self, path: str):
        """Сохраняет модель."""
        torch.save({
            "q_network": self.q_network.state_dict(),
            "q_target": self.q_target.state_dict(),
            "mixer": self.mixer.state_dict(),
            "mixer_target": self.mixer_target.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "train_step": self.train_step,
        }, path)

    def load(self, path: str):
        """Загружает модель."""
        checkpoint = torch.load(path, map_location=self.device)
        self.q_network.load_state_dict(checkpoint["q_network"])
        self.q_target.load_state_dict(checkpoint["q_target"])
        self.mixer.load_state_dict(checkpoint["mixer"])
        self.mixer_target.load_state_dict(checkpoint["mixer_target"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.train_step = checkpoint["train_step"]
