import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from config import SEED


class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return self.softmax(x)


def reinforce(env, num_episodes=3000, gamma=0.99, lr=1e-3, hidden_dim=128, print_every=200):
    input_dim = env.observation_space.n # 500
    output_dim = env.action_space.n     # 6

    policy_net = PolicyNetwork(input_dim, hidden_dim, output_dim)
    optimizer = optim.Adam(policy_net.parameters(), lr=lr)

    episode_rewards = []

    def preprocess_state(state):
        one_hot = np.zeros(input_dim, dtype=np.float32)
        one_hot[state] = 1.0
        return torch.tensor(one_hot)

    for i_episode in range(num_episodes):
        state, info = env.reset(seed=SEED + i_episode)
        log_probs, rewards = [], []
        done = False

        while not done:
            state_tensor = preprocess_state(state)
            probs = policy_net(state_tensor)
            action = torch.multinomial(probs, 1).item()
            log_prob = torch.log(probs[action])

            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            log_probs.append(log_prob)
            rewards.append(reward)
            state = next_state

        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + gamma * G
            returns.insert(0, G)
        returns = torch.tensor(returns, dtype=torch.float32)

        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        policy_loss = torch.stack([-lp * G for lp, G in zip(log_probs, returns)]).sum()

        optimizer.zero_grad()
        policy_loss.backward()
        optimizer.step()

        total_reward = sum(rewards)
        episode_rewards.append(total_reward)

        if (i_episode + 1) % print_every == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            print(f"REINFORCE | Ep {i_episode+1:4d} | AvgR(last 100): {avg_reward:.2f}")

    return policy_net, episode_rewards

