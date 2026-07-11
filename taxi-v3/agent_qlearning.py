import numpy as np
from collections import defaultdict
from config import SEED


def q_learning(env, num_episodes=2000, alpha=0.1, gamma=0.99,
               epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.9995):
    n_actions = env.action_space.n
    Q = defaultdict(lambda: np.zeros(n_actions))
    epsilon = epsilon_start
    episode_rewards = []

    for i_episode in range(num_episodes):
        state, info = env.reset(seed=SEED + i_episode)
        total_reward = 0
        done = False

        while not done:
            if np.random.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(Q[state])

            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            best_next_action = np.argmax(Q[next_state])
            td_target = reward + gamma * Q[next_state][best_next_action] * (not terminated)
            td_error = td_target - Q[state][action]
            Q[state][action] += alpha * td_error

            state = next_state
            total_reward += reward

        episode_rewards.append(total_reward)
        epsilon = max(epsilon_end, epsilon * epsilon_decay)

        if (i_episode + 1) % 200 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            print(f"Q-Learning | Ep {i_episode+1:4d} | AvgR(last 100): {avg_reward:.2f} | ε: {epsilon:.3f}")

    return Q, episode_rewards