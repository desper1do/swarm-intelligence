import time
import numpy as np
import torch


def play_agent(env, Q, episodes=3, delay=0.3):
    for ep in range(episodes):
        state, info = env.reset()
        total_reward = 0
        done = False

        while not done:
            env.render()      # отрисовка текущего кадра в окне
            time.sleep(delay) # задержка для восприятия человеком
            action = np.argmax  (Q[state])
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
            state = next_state

        print(f"Demo episode {ep+1} finished with reward: {total_reward}")

    env.close()


def play_reinforce(env, policy_net, episodes=3, delay=0.3):
    input_dim = env.observation_space.n

    for ep in range(episodes):
        state, info = env.reset()
        total_reward = 0
        done = False

        while not done:
            env.render()
            time.sleep(delay)
            state_tensor = torch.tensor(np.eye(input_dim)[state], dtype=torch.float32)
            probs = policy_net(state_tensor)
            action = torch.argmax(probs).item()
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
            state = next_state

        print(f"Demo episode {ep+1} finished with reward: {total_reward}")

    env.close()