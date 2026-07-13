"""
Replay Buffer для MARL алгоритмов.
Поддержка обычного буфера (MADDPG) и эпизодического (QMIX).
"""

import numpy as np
import torch
from typing import Dict, List, Optional, Tuple
from collections import deque


class ReplayBuffer:
    """
    Стандартный Replay Buffer для off-policy алгоритмов (MADDPG).
    Хранит переходы (s, o, a, r, s', o', done) для всех агентов.
    """

    def __init__(
        self,
        capacity: int,
        n_agents: int,
        obs_dim: int,
        action_dim: int,
        state_dim: int,
        device: str = "cuda",
    ):
        self.capacity = capacity
        self.n_agents = n_agents
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.device = device

        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.observations = np.zeros((capacity, n_agents, obs_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, n_agents, action_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.next_observations = np.zeros((capacity, n_agents, obs_dim), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.float32)

        self.ptr = 0
        self.size = 0

    def add(
        self,
        state: np.ndarray,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        next_obs: np.ndarray,
        done: bool,
    ):
        """Добавляет переход в буфер."""
        self.states[self.ptr] = state
        self.observations[self.ptr] = obs
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_states[self.ptr] = next_state
        self.next_observations[self.ptr] = next_obs
        self.dones[self.ptr] = done

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """Сэмплирует батч из буфера."""
        indices = np.random.randint(0, self.size, size=batch_size)

        batch = {
            "states": torch.FloatTensor(self.states[indices]).to(self.device),
            "observations": torch.FloatTensor(self.observations[indices]).to(self.device),
            "actions": torch.FloatTensor(self.actions[indices]).to(self.device),
            "rewards": torch.FloatTensor(self.rewards[indices]).to(self.device),
            "next_states": torch.FloatTensor(self.next_states[indices]).to(self.device),
            "next_observations": torch.FloatTensor(self.next_observations[indices]).to(self.device),
            "dones": torch.FloatTensor(self.dones[indices]).to(self.device),
        }
        return batch

    def __len__(self):
        return self.size


class EpisodeBuffer:
    """
    Буфер для хранения эпизодов (для QMIX).
    Хранит полные эпизоды для обучения с учётом последовательности.
    """

    def __init__(
        self,
        capacity: int,
        n_agents: int,
        obs_dim: int,
        state_dim: int,
        n_actions: int,
        episode_limit: int,
        device: str = "cuda",
    ):
        self.capacity = capacity
        self.n_agents = n_agents
        self.obs_dim = obs_dim
        self.state_dim = state_dim
        self.n_actions = n_actions
        self.episode_limit = episode_limit
        self.device = device

        # Буфер эпизодов
        self.episodes = deque(maxlen=capacity)

        # Текущий эпизод
        self._reset_current_episode()

    def _reset_current_episode(self):
        """Сбрасывает текущий эпизод."""
        self.curr_obs = []
        self.curr_states = []
        self.curr_actions = []
        self.curr_rewards = []
        self.curr_next_obs = []
        self.curr_next_states = []
        self.curr_dones = []
        self.curr_avail_actions = []
        self.curr_next_avail_actions = []
        self.curr_padded = []

    def add_step(
        self,
        obs: np.ndarray,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_obs: np.ndarray,
        next_state: np.ndarray,
        done: bool,
        avail_actions: np.ndarray,
        next_avail_actions: np.ndarray,
        padded: bool = False,
    ):
        """Добавляет шаг текущего эпизода."""
        self.curr_obs.append(obs)
        self.curr_states.append(state)
        self.curr_actions.append(action)
        self.curr_rewards.append(reward)
        self.curr_next_obs.append(next_obs)
        self.curr_next_states.append(next_state)
        self.curr_dones.append(done)
        self.curr_avail_actions.append(avail_actions)
        self.curr_next_avail_actions.append(next_avail_actions)
        self.curr_padded.append(padded)

    def end_episode(self):
        """Завершает текущий эпизод и сохраняет его."""
        if len(self.curr_obs) == 0:
            return

        episode = {
            "obs": np.array(self.curr_obs, dtype=np.float32),
            "states": np.array(self.curr_states, dtype=np.float32),
            "actions": np.array(self.curr_actions, dtype=np.int64),
            "rewards": np.array(self.curr_rewards, dtype=np.float32),
            "next_obs": np.array(self.curr_next_obs, dtype=np.float32),
            "next_states": np.array(self.curr_next_states, dtype=np.float32),
            "dones": np.array(self.curr_dones, dtype=np.float32),
            "avail_actions": np.array(self.curr_avail_actions, dtype=np.float32),
            "next_avail_actions": np.array(self.curr_next_avail_actions, dtype=np.float32),
            "padded": np.array(self.curr_padded, dtype=np.float32),
        }

        self.episodes.append(episode)
        self._reset_current_episode()

    def sample(self, batch_size: int) -> Optional[Dict[str, torch.Tensor]]:
        """Сэмплирует батч эпизодов."""
        if len(self.episodes) < batch_size:
            return None

        indices = np.random.choice(len(self.episodes), size=batch_size, replace=False)

        # Находим максимальную длину эпизода в выборке
        max_len = max(len(self.episodes[i]["obs"]) for i in indices)

        # Паддинг эпизодов до одинаковой длины
        batch_obs = []
        batch_states = []
        batch_actions = []
        batch_rewards = []
        batch_next_obs = []
        batch_next_states = []
        batch_dones = []
        batch_avail_actions = []
        batch_next_avail_actions = []
        batch_padded = []
        batch_mask = []

        for i in indices:
            ep = self.episodes[i]
            ep_len = len(ep["obs"])
            pad_len = max_len - ep_len

            batch_obs.append(self._pad(ep["obs"], pad_len))
            batch_states.append(self._pad(ep["states"], pad_len))
            batch_actions.append(self._pad(ep["actions"], pad_len))
            batch_rewards.append(self._pad(ep["rewards"].reshape(-1, 1), pad_len))
            batch_next_obs.append(self._pad(ep["next_obs"], pad_len))
            batch_next_states.append(self._pad(ep["next_states"], pad_len))
            batch_dones.append(self._pad(ep["dones"].reshape(-1, 1), pad_len))
            batch_avail_actions.append(self._pad(ep["avail_actions"], pad_len))
            batch_next_avail_actions.append(self._pad(ep["next_avail_actions"], pad_len))
            batch_padded.append(self._pad(ep["padded"].reshape(-1, 1), pad_len))

            # Маска для реальных шагов (не паддинга)
            mask = np.concatenate([
                np.ones((ep_len, 1), dtype=np.float32),
                np.zeros((pad_len, 1), dtype=np.float32),
            ])
            batch_mask.append(mask)

        return {
            "obs": torch.FloatTensor(np.stack(batch_obs)).to(self.device),
            "states": torch.FloatTensor(np.stack(batch_states)).to(self.device),
            "actions": torch.LongTensor(np.stack(batch_actions)).to(self.device),
            "rewards": torch.FloatTensor(np.stack(batch_rewards)).to(self.device),
            "next_obs": torch.FloatTensor(np.stack(batch_next_obs)).to(self.device),
            "next_states": torch.FloatTensor(np.stack(batch_next_states)).to(self.device),
            "dones": torch.FloatTensor(np.stack(batch_dones)).to(self.device),
            "avail_actions": torch.FloatTensor(np.stack(batch_avail_actions)).to(self.device),
            "next_avail_actions": torch.FloatTensor(np.stack(batch_next_avail_actions)).to(self.device),
            "padded": torch.FloatTensor(np.stack(batch_padded)).to(self.device),
            "mask": torch.FloatTensor(np.stack(batch_mask)).to(self.device),
        }

    def _pad(self, arr: np.ndarray, pad_len: int) -> np.ndarray:
        """Дополняет массив нулями до нужной длины."""
        if pad_len == 0:
            return arr
        pad_shape = (pad_len,) + arr.shape[1:]
        pad = np.zeros(pad_shape, dtype=arr.dtype)
        return np.concatenate([arr, pad], axis=0)

    def __len__(self):
        return len(self.episodes)


class RolloutBuffer:
    """
    Буфер для on-policy алгоритмов (MAPPO).
    Хранит траектории для обновления политики.
    """

    def __init__(
        self,
        n_steps: int,
        n_agents: int,
        obs_dim: int,
        state_dim: int,
        action_dim: int,
        n_envs: int = 1,
        device: str = "cuda",
    ):
        self.n_steps = n_steps
        self.n_agents = n_agents
        self.obs_dim = obs_dim
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.n_envs = n_envs
        self.device = device

        self.observations = np.zeros((n_steps, n_envs, n_agents, obs_dim), dtype=np.float32)
        self.states = np.zeros((n_steps, n_envs, state_dim), dtype=np.float32)
        self.actions = np.zeros((n_steps, n_envs, n_agents), dtype=np.int64)
        self.log_probs = np.zeros((n_steps, n_envs, n_agents), dtype=np.float32)
        self.rewards = np.zeros((n_steps, n_envs, 1), dtype=np.float32)
        self.values = np.zeros((n_steps, n_envs, 1), dtype=np.float32)
        self.dones = np.zeros((n_steps, n_envs, 1), dtype=np.float32)
        self.avail_actions = np.zeros((n_steps, n_envs, n_agents, action_dim), dtype=np.float32)

        self.ptr = 0
        self.size = 0

    def add(
        self,
        obs: np.ndarray,
        state: np.ndarray,
        action: np.ndarray,
        log_prob: np.ndarray,
        reward: float,
        value: np.ndarray,
        done: bool,
        avail_actions: np.ndarray,
    ):
        """Добавляет шаг в буфер."""
        idx = self.ptr % self.n_steps
        self.observations[idx] = obs
        self.states[idx] = state
        self.actions[idx] = action
        self.log_probs[idx] = log_prob
        self.rewards[idx] = reward
        self.values[idx] = value
        self.dones[idx] = done
        self.avail_actions[idx] = avail_actions

        self.ptr += 1
        self.size = min(self.size + 1, self.n_steps)

    def compute_returns_and_advantages(
        self,
        next_value: np.ndarray,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Вычисляет returns и advantages используя GAE.

        Args:
            next_value: (n_envs, 1) - значение следующего состояния
            gamma: дисконт-фактор
            gae_lambda: параметр GAE

        Returns:
            returns: (size, n_envs, 1)
            advantages: (size, n_envs, 1)
        """
        returns = np.zeros_like(self.rewards[:self.size])
        advantages = np.zeros_like(self.rewards[:self.size])

        last_gae = 0
        for t in reversed(range(self.size)):
            if t == self.size - 1:
                next_val = next_value
            else:
                next_val = self.values[t + 1]

            delta = (
                self.rewards[t]
                + gamma * next_val * (1 - self.dones[t])
                - self.values[t]
            )
            last_gae = delta + gamma * gae_lambda * (1 - self.dones[t]) * last_gae
            advantages[t] = last_gae
            returns[t] = advantages[t] + self.values[t]

        return returns, advantages

    def get_batches(self, batch_size: int):
        """Возвращает батчи данных."""
        flat_obs = self.observations[:self.size].reshape(-1, self.n_agents, self.obs_dim)
        flat_states = self.states[:self.size].reshape(-1, self.state_dim)
        flat_actions = self.actions[:self.size].reshape(-1, self.n_agents)
        flat_log_probs = self.log_probs[:self.size].reshape(-1, self.n_agents)
        flat_avail_actions = self.avail_actions[:self.size].reshape(-1, self.n_agents, self.action_dim)

        total_size = flat_obs.shape[0]
        indices = np.arange(total_size)
        np.random.shuffle(indices)

        for start in range(0, total_size, batch_size):
            end = min(start + batch_size, total_size)
            batch_idx = indices[start:end]

            yield {
                "obs": torch.FloatTensor(flat_obs[batch_idx]).to(self.device),
                "states": torch.FloatTensor(flat_states[batch_idx]).to(self.device),
                "actions": torch.LongTensor(flat_actions[batch_idx]).to(self.device),
                "old_log_probs": torch.FloatTensor(flat_log_probs[batch_idx]).to(self.device),
                "avail_actions": torch.FloatTensor(flat_avail_actions[batch_idx]).to(self.device),
            }

    def clear(self):
        """Очищает буфер."""
        self.ptr = 0
        self.size = 0

    def is_full(self) -> bool:
        return self.size >= self.n_steps

    def __len__(self):
        return self.size
