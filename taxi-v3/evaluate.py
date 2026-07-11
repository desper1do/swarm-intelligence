import numpy as np
import torch


def evaluate_agent(env, Q, num_episodes=100, seed=123):
    np.random.seed(seed)
    total_rewards = []

    for i in range(num_episodes):
        state, info = env.reset(seed=seed + i)
        total_reward = 0
        done = False

        while not done:
            action = np.argmax(Q[state])
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
            state = next_state

        total_rewards.append(total_reward)

    return np.mean(total_rewards), total_rewards


def evaluate_reinforce(env, policy_net, num_episodes=100, seed=123):
    input_dim = env.observation_space.n
    total_rewards = []

    for i in range(num_episodes):
        state, info = env.reset(seed=seed + i)
        total_reward = 0
        done = False

        while not done:
            state_tensor = torch.tensor(np.eye(input_dim)[state], dtype=torch.float32)
            probs = policy_net(state_tensor)
            action = torch.argmax(probs).item()
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
            state = next_state

        total_rewards.append(total_reward)

    return np.mean(total_rewards), total_rewards