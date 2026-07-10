# agent_maddpg.py
import numpy as np
import torch
from networks_maddpg import Actor, Critic
from config import MADDPG


class MADDPGAgent:
    # шаг 1 плана MADDPG: для каждого из n_agents - свой актер, свой критик и их
    # целевые копии. Критики централизованы (видят всех), актеры - нет
    def __init__(self, obs_shape, n_actions, n_agents, device):
        self.n_agents = n_agents
        self.n_actions = n_actions
        self.device = device
        self.sigma = MADDPG.SIGMA

        self.actors = [Actor(obs_shape, n_actions, MADDPG.HIDDEN_SIZE).to(device) for _ in range(n_agents)]
        self.target_actors = [Actor(obs_shape, n_actions, MADDPG.HIDDEN_SIZE).to(device) for _ in range(n_agents)]
        for a, ta in zip(self.actors, self.target_actors):
            ta.load_state_dict(a.state_dict())

        self.critics = [Critic(obs_shape, n_actions, n_agents, MADDPG.HIDDEN_SIZE).to(device) for _ in range(n_agents)]
        self.target_critics = [Critic(obs_shape, n_actions, n_agents, MADDPG.HIDDEN_SIZE).to(device) for _ in range(n_agents)]
        for c, tc in zip(self.critics, self.target_critics):
            tc.load_state_dict(c.state_dict())

        self.actor_optimizers = [torch.optim.Adam(a.parameters(), lr=MADDPG.LR_ACTOR) for a in self.actors]
        self.critic_optimizers = [torch.optim.Adam(c.parameters(), lr=MADDPG.LR_CRITIC) for c in self.critics]

        self._update_count = 0  # счетчик апдейтов критика для отложенного обновления актера (TD3)

    def choose_actions(self, obs_list, explore=True):
        # шаг 3 плана MADDPG: a_i = mu_i(o_i) + шум, дискретизация через argmax
        obs = torch.as_tensor(np.stack(obs_list), dtype=torch.float32, device=self.device)
        actions = []
        with torch.no_grad():
            for i in range(self.n_agents):
                logits = self.actors[i](obs[i:i + 1])
                if explore:
                    noise = torch.clamp(
                        torch.randn_like(logits) * self.sigma, -MADDPG.NOISE_CLIP, MADDPG.NOISE_CLIP
                    )
                    logits = logits + noise
                actions.append(int(logits.argmax(dim=-1).item()))
        return actions

    def decay_sigma(self):
        self.sigma = max(MADDPG.SIGMA_MIN, self.sigma * MADDPG.SIGMA_DECAY)

    def soft_update(self):
        # шаг 9 плана MADDPG: theta' = tau*theta + (1-tau)*theta'
        for i in range(self.n_agents):
            for tp, p in zip(self.target_actors[i].parameters(), self.actors[i].parameters()):
                tp.data.copy_(MADDPG.TAU * p.data + (1 - MADDPG.TAU) * tp.data)
            for tp, p in zip(self.target_critics[i].parameters(), self.critics[i].parameters()):
                tp.data.copy_(MADDPG.TAU * p.data + (1 - MADDPG.TAU) * tp.data)

    def snapshot(self):
        # копия весов всех актеров (best-checkpoint): для финальных графиков/демо
        # берем лучшую по success_rate политику, а не последнюю
        import copy
        return {"actors": [copy.deepcopy(a.state_dict()) for a in self.actors]}

    def load_snapshot(self, snap):
        for a, sd in zip(self.actors, snap["actors"]):
            a.load_state_dict(sd)
