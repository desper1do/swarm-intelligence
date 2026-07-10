# networks_maddpg.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    # общий сверточный энкодер локального наблюдения, используется и в актере,
    # и в критике (архитектурно общий, веса у каждой сети свои)
    def __init__(self, obs_shape, embed_dim):
        super().__init__()
        c, h, w = obs_shape
        self.conv = nn.Sequential(
            nn.Conv2d(c, 16, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, padding=1), nn.ReLU(),
        )
        self.fc = nn.Linear(16 * h * w, embed_dim)

    def forward(self, obs):
        x = self.conv(obs)
        x = x.reshape(x.size(0), -1)
        return F.relu(self.fc(x))


class Actor(nn.Module):
    # актер mu_i(o_i) (шаг 1 плана MADDPG) - децентрализован, видит только
    # свое локальное наблюдение. Выход - логиты по 5 дискретным действиям
    # (непрерывный выход DDPG адаптирован под дискретное пространство POGEMA
    # через Gumbel-Softmax при обучении, см. train_maddpg.py)
    def __init__(self, obs_shape, n_actions, hidden_size):
        super().__init__()
        self.encoder = Encoder(obs_shape, hidden_size)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, n_actions),
        )

    def forward(self, obs):
        return self.fc(self.encoder(obs))  # логиты, без softmax - его накладывают снаружи


class Critic(nn.Module):
    # централизованный критик Q_i(o_1..o_n, a_1..a_n) (шаг 1 плана MADDPG) - при
    # обучении видит наблюдения и действия ВСЕХ агентов (CTDE), не только своего.
    # TD3-стиль: две независимые Q-головы (twin critics) со своими энкодерами.
    # цель строится по min(Q1, Q2) - это гасит переоценку Q, из-за которой
    # ванильный MADDPG (одиночный критик) коллапсирует на длинной дистанции
    def __init__(self, obs_shape, n_actions, n_agents, hidden_size):
        super().__init__()
        embed_dim = hidden_size // 2
        in_dim = n_agents * embed_dim + n_agents * n_actions

        def make_branch():
            return nn.Sequential(
                nn.Linear(in_dim, hidden_size), nn.ReLU(),
                nn.Linear(hidden_size, hidden_size), nn.ReLU(),
                nn.Linear(hidden_size, 1),
            )

        self.encoder1 = Encoder(obs_shape, embed_dim)
        self.encoder2 = Encoder(obs_shape, embed_dim)
        self.q1_head = make_branch()
        self.q2_head = make_branch()

    def _features(self, encoder, all_obs, all_actions_onehot):
        batch, n_agents = all_obs.shape[0], all_obs.shape[1]
        obs_flat = all_obs.reshape(batch * n_agents, *all_obs.shape[2:])
        embeds = encoder(obs_flat).view(batch, n_agents, -1)
        return torch.cat([embeds.reshape(batch, -1), all_actions_onehot.reshape(batch, -1)], dim=-1)

    def forward(self, all_obs, all_actions_onehot):
        # обе Q-оценки (для обучения критика)
        q1 = self.q1_head(self._features(self.encoder1, all_obs, all_actions_onehot)).squeeze(-1)
        q2 = self.q2_head(self._features(self.encoder2, all_obs, all_actions_onehot)).squeeze(-1)
        return q1, q2

    def q1(self, all_obs, all_actions_onehot):
        # только первая Q-голова (для градиента политики актера)
        return self.q1_head(self._features(self.encoder1, all_obs, all_actions_onehot)).squeeze(-1)
