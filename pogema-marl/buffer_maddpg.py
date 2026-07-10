# buffer_maddpg.py
import numpy as np
import random
from collections import deque


class ReplayBuffer:
    # обычный буфер отдельных переходов (шаг 5 плана MADDPG) - в отличие от QMIX
    # тут нет RNN, каждый переход независим, поэтому эпизоды хранить целиком не нужно
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def add(self, obs, actions, rewards, next_obs, done):
        # obs/next_obs: список из n_agents наблюдений, actions/rewards: списки по агентам
        self.buffer.append((obs, actions, rewards, next_obs, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        obs, actions, rewards, next_obs, done = zip(*batch)
        return (
            np.array(obs, dtype=np.float32),        # (B, n_agents, 3, 7, 7)
            np.array(actions, dtype=np.int64),       # (B, n_agents)
            np.array(rewards, dtype=np.float32),     # (B, n_agents)
            np.array(next_obs, dtype=np.float32),    # (B, n_agents, 3, 7, 7)
            np.array(done, dtype=np.float32),        # (B, n_agents)
        )
