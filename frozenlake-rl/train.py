import numpy as np 
from config import N_EPISODES, MOVING_AVG_WINDOW
from environment import make_env, get_env_info

def train(agent, n_episodes=N_EPISODES, verbose=True, seed=None):
    # цикл обучения Q-Learning (шаги 2-8 из плана задания)
    env = make_env()
    rewards_history = []
    epsilon_history = []

    for episode in range(n_episodes):
        # Шаг 2: начало эпизода, сброс среды.
        # seed передаем только в самый первый reset - дальше среда сама
        # детерминированно раскручивает свой генератор случайных чисел
        # (в т.ч. те самые "скольжения" из is_slippery=True)
        state, info = env.reset(seed=seed) if episode == 0 else env.reset()
        done = False
        total_reward = 0

        # Шаги 3-6: выбор действия, шаг среды, обновление Q, переход s -> s'
        while not(done):
            action = agent.choose_action(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            agent.update(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

        agent.decay_epsilon()
        rewards_history.append(total_reward)
        epsilon_history.append(agent.epsilon)

        if verbose and (episode + 1) % 1000 == 0:
            avg_reward = np.mean(rewards_history[-MOVING_AVG_WINDOW:])
            print(f"Episode {episode + 1}/{n_episodes}, Average Reward: {avg_reward:.3f}, Epsilon: {agent.epsilon:.3f}")

    env.close()
    return rewards_history, epsilon_history


def train_sarsa(agent, n_episodes=N_EPISODES, verbose=True, seed=None):
    # цикл обучения SARSA (шаги 2-8 из плана задания)
    env = make_env()
    rewards_history = []
    epsilon_history = []

    for episode in range(n_episodes):
        # Шаг 2: начало эпизода - сброс среды и выбор первого действия a
        state, info = env.reset(seed=seed) if episode == 0 else env.reset()
        done = False
        total_reward = 0
        action = agent.choose_action(state)

        # Шаги 3-6: выполнение a, выбор a' для s' (Шаг 4), обновление Q по Q(s',a'),
        # затем перенос s=s', a=a' - в этом отличие SARSA от Q-Learning
        while not (done):
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            next_action = agent.choose_action(next_state)
            agent.update(state, action, reward, next_state, next_action, done)
            state = next_state
            action = next_action
            total_reward += reward

        agent.decay_epsilon()
        rewards_history.append(total_reward)
        epsilon_history.append(agent.epsilon)

        if verbose and (episode + 1) % 1000 == 0:
            avg_reward = np.mean(rewards_history[-MOVING_AVG_WINDOW:])
            print(f"Episode {episode + 1}/{n_episodes}, Average Reward: {avg_reward:.3f}, Epsilon: {agent.epsilon:.3f}")

    env.close()
    return rewards_history, epsilon_history