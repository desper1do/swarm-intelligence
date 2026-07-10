# buffer_qmix.py
import numpy as np
from collections import deque
import random


class EpisodeBuffer:
    # буфер воспроизведения QMIX хранит ЦЕЛЫЕ эпизоды (не отдельные переходы),
    # т.к. агентские сети - DRQN с GRU и обучаются через BPTT по всей траектории
    # (шаг 5 плана QMIX). Каждый эпизод дополняется нулями до max_episode_steps,
    # filled_mask отмечает реальные шаги, чтобы паддинг не влиял на loss
    def __init__(self, capacity, max_episode_steps, n_agents, obs_shape, state_dim):
        self.capacity = capacity
        self.max_episode_steps = max_episode_steps
        self.n_agents = n_agents
        self.obs_shape = obs_shape
        self.state_dim = state_dim
        self.episodes = deque(maxlen=capacity)

    def __len__(self):
        return len(self.episodes)

    def start_episode(self):
        # награда в QMIX общая (team_reward), т.к. вся идея факторизации Q_tot
        # исходит из единой совместной функции ценности - поэтому по агентам не
        # делится, отдельная avail-маска по агентам не нужна: даже завершившие
        # маршрут агенты продолжают участвовать в Q_tot до конца эпизода (их
        # наблюдение просто замораживается), а padding после конца эпизода
        # отсекается маской filled
        T, n, obs_shape = self.max_episode_steps, self.n_agents, self.obs_shape
        return {
            "obs": np.zeros((T, n, *obs_shape), dtype=np.float32),
            "state": np.zeros((T, self.state_dim), dtype=np.float32),
            "actions": np.zeros((T, n), dtype=np.int64),
            "team_reward": np.zeros((T,), dtype=np.float32),
            "done": np.zeros((T,), dtype=np.float32),  # последний реальный шаг эпизода
            "filled": np.zeros((T,), dtype=np.float32),
        }

    def add_episode(self, episode):
        self.episodes.append(episode)

    def sample(self, batch_size):
        batch = random.sample(self.episodes, min(batch_size, len(self.episodes)))
        out = {key: np.stack([ep[key] for ep in batch]) for key in batch[0]}
        return out
