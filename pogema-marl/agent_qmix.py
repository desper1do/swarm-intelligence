# agent_qmix.py
import numpy as np
import torch
from networks_qmix import AgentNetwork, MixingNetwork
from config import QMIX


class QMIXAgent:
    # обертка над агентской + смешивающей сетью (шаг 1 плана QMIX): хранит
    # основные и целевые сети, отвечает за выбор действий (epsilon-greedy по
    # своему Q_i - шаг 2) и soft update целевых сетей (шаг 8)
    def __init__(self, obs_shape, n_actions, n_agents, state_dim, device):
        self.n_agents = n_agents
        self.n_actions = n_actions
        self.device = device
        self.epsilon = QMIX.EPSILON_START

        self.agent_net = AgentNetwork(obs_shape, n_actions, n_agents, QMIX.HIDDEN_SIZE).to(device)
        self.target_agent_net = AgentNetwork(obs_shape, n_actions, n_agents, QMIX.HIDDEN_SIZE).to(device)
        self.target_agent_net.load_state_dict(self.agent_net.state_dict())

        self.mixer = MixingNetwork(n_agents, state_dim, QMIX.MIXING_HIDDEN_SIZE).to(device)
        self.target_mixer = MixingNetwork(n_agents, state_dim, QMIX.MIXING_HIDDEN_SIZE).to(device)
        self.target_mixer.load_state_dict(self.mixer.state_dict())

        params = list(self.agent_net.parameters()) + list(self.mixer.parameters())
        self.optimizer = torch.optim.Adam(params, lr=QMIX.LR)

        self._agent_id_onehot = torch.eye(n_agents, device=device)

    def init_hidden(self):
        # одно скрытое состояние GRU на агента (батч = n_agents, эпизод один)
        return self.agent_net.init_hidden(self.n_agents, self.device)

    def choose_actions(self, obs_list, hidden):
        # obs_list: n_agents наблюдений за текущий шаг, hidden: (n_agents, hidden_size)
        obs = torch.as_tensor(np.stack(obs_list), dtype=torch.float32, device=self.device)
        with torch.no_grad():
            q_values, hidden = self.agent_net(obs, self._agent_id_onehot, hidden)
        actions = []
        for i in range(self.n_agents):
            # epsilon > 0 проверяем ДО вызова np.random.rand(): иначе даже при
            # epsilon=0.0 (жадный прогон для демонстрации/визуализации, вне
            # обучения) метод дергает общий генератор numpy впустую и сдвигает
            # последовательность случайных чисел, которую использует настоящее
            # обучение - ломает воспроизводимость и де-факто портит exploration
            if self.epsilon > 0 and np.random.rand() < self.epsilon:
                actions.append(np.random.randint(self.n_actions))
            else:
                actions.append(int(q_values[i].argmax().item()))
        return actions, hidden

    def decay_epsilon(self):
        self.epsilon = max(QMIX.EPSILON_MIN, self.epsilon * QMIX.EPSILON_DECAY)

    def soft_update(self):
        # шаг 8 плана QMIX: polyak averaging целевых сетей
        for tp, p in zip(self.target_agent_net.parameters(), self.agent_net.parameters()):
            tp.data.copy_(QMIX.TAU * p.data + (1 - QMIX.TAU) * tp.data)
        for tp, p in zip(self.target_mixer.parameters(), self.mixer.parameters()):
            tp.data.copy_(QMIX.TAU * p.data + (1 - QMIX.TAU) * tp.data)

    def snapshot(self):
        # копия весов агентской + смешивающей сети (best-checkpoint): даже у
        # стабилизированного обучения бывают просадки, поэтому для финальных
        # графиков/демо берем веса лучшего эпизода, а не последнего
        import copy
        return {
            "agent_net": copy.deepcopy(self.agent_net.state_dict()),
            "mixer": copy.deepcopy(self.mixer.state_dict()),
        }

    def load_snapshot(self, snap):
        self.agent_net.load_state_dict(snap["agent_net"])
        self.mixer.load_state_dict(snap["mixer"])
