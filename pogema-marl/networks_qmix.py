# networks_qmix.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class AgentNetwork(nn.Module):
    # DRQN одного агента (шаг 1 плана QMIX): CNN по локальному наблюдению -> GRU
    # (память по траектории tau) -> Q-значения по действиям. Сеть одна на всех
    # агентов (параметры общие, как в референсной реализации QMIX/PyMARL) -
    # чтобы агенты не были при этом идентичны, на вход добавляется one-hot id агента
    def __init__(self, obs_shape, n_actions, n_agents, hidden_size):
        super().__init__()
        c, h, w = obs_shape
        self.hidden_size = hidden_size

        self.conv = nn.Sequential(
            nn.Conv2d(c, 16, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, padding=1), nn.ReLU(),
        )
        conv_out = 16 * h * w
        self.fc_in = nn.Linear(conv_out + n_agents, hidden_size)
        self.gru = nn.GRUCell(hidden_size, hidden_size)
        self.fc_out = nn.Linear(hidden_size, n_actions)

    def init_hidden(self, batch_size, device):
        return torch.zeros(batch_size, self.hidden_size, device=device)

    def forward(self, obs, agent_id_onehot, hidden):
        # obs: (batch, c, h, w), agent_id_onehot: (batch, n_agents), hidden: (batch, hidden_size)
        x = self.conv(obs)
        x = x.reshape(x.size(0), -1)
        x = torch.cat([x, agent_id_onehot], dim=-1)
        x = F.relu(self.fc_in(x))
        hidden = self.gru(x, hidden)
        q_values = self.fc_out(hidden)
        return q_values, hidden


class MixingNetwork(nn.Module):
    # смешивающая сеть (шаг 1 плана QMIX): объединяет индивидуальные Q_i в Q_tot.
    # веса генерируются гиперсетями из глобального состояния s и берутся по модулю
    # (abs) - это и обеспечивает монотонность dQ_tot/dQ_i >= 0 (п.2.1 задания)
    def __init__(self, n_agents, state_dim, mixing_hidden_size):
        super().__init__()
        self.n_agents = n_agents
        self.mixing_hidden_size = mixing_hidden_size

        self.hyper_w1 = nn.Linear(state_dim, n_agents * mixing_hidden_size)
        self.hyper_b1 = nn.Linear(state_dim, mixing_hidden_size)

        self.hyper_w2 = nn.Linear(state_dim, mixing_hidden_size)
        self.hyper_b2 = nn.Sequential(
            nn.Linear(state_dim, mixing_hidden_size), nn.ReLU(),
            nn.Linear(mixing_hidden_size, 1),
        )

    def forward(self, agent_qs, state):
        # agent_qs: (batch, n_agents) - Q-значения выбранных действий каждого агента
        # state: (batch, state_dim) - глобальное состояние
        batch_size = agent_qs.size(0)
        agent_qs = agent_qs.view(batch_size, 1, self.n_agents)

        w1 = torch.abs(self.hyper_w1(state)).view(batch_size, self.n_agents, self.mixing_hidden_size)
        b1 = self.hyper_b1(state).view(batch_size, 1, self.mixing_hidden_size)
        hidden = F.elu(torch.bmm(agent_qs, w1) + b1)

        w2 = torch.abs(self.hyper_w2(state)).view(batch_size, self.mixing_hidden_size, 1)
        b2 = self.hyper_b2(state).view(batch_size, 1, 1)
        q_tot = torch.bmm(hidden, w2) + b2

        return q_tot.view(batch_size)
