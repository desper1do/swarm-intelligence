# main.py
import os
import numpy as np

from config import N_EPISODES, MOVING_AVG_WINDOW, SEED
from environment import make_env, get_env_info
from agent_qlearning import QLearningAgent
from agent_sarsa import SARSAAgent
from train import train, train_sarsa
from animate import animate_training, run_episode_qlearning, run_episode_sarsa
from visualize import plot_convergence, plot_epsilon, plot_q_heatmap, plot_policy
from compare import plot_comparison, plot_policy_diff
from demo import demo_episode

PLOTS_DIR = os.path.join(os.path.dirname(__file__), "plots")


def print_policy(q_table, n_states):
    # Быстрый текстовый вывод политики (по одной стрелке на клетку)
    arrows = {0: "←", 1: "↓", 2: "→", 3: "↑"}
    side = int(np.sqrt(n_states))  # 4 для карты 4x4
    policy = [arrows[np.argmax(q_table[s])] for s in range(n_states)]
    for row in range(side):
        print(" ".join(policy[row * side:(row + 1) * side]))


if __name__ == "__main__":
    os.makedirs(PLOTS_DIR, exist_ok=True)

    env = make_env()
    n_states, n_actions = get_env_info(env)
    env.close()

    # анимация обучения в реальном времени (п.4.1 задания, обновление Q-таблицы "на лету")
    print("Анимация обучения Q-Learning (закройте окно, чтобы продолжить)")
    np.random.seed(SEED)
    animate_training(QLearningAgent(n_states, n_actions), run_episode_qlearning, label="Q-Learning", color="blue")

    print("Анимация обучения SARSA (закройте окно, чтобы продолжить)")
    np.random.seed(SEED)
    animate_training(SARSAAgent(n_states, n_actions), run_episode_sarsa, label="SARSA", color="green")

    # полное обучение (без анимации) для итоговых метрик и графиков.
    # np.random.seed фиксирует и epsilon-greedy случайность агента, и (через
    # seed= в train/train_sarsa) случайность скольжения в самой среде -
    # без этого при неудачном стечении обстоятельств Q-таблица может
    # ни разу не получить ненулевую награду за все обучение (см. config.py)
    print("\nОбучение Q-Learning")
    np.random.seed(SEED)
    ql_agent = QLearningAgent(n_states, n_actions)
    ql_rewards, ql_epsilons = train(ql_agent, n_episodes=N_EPISODES, seed=SEED)

    final_avg_ql = np.mean(ql_rewards[-MOVING_AVG_WINDOW:])
    print(f"\nQ-Learning: средняя награда за последние {MOVING_AVG_WINDOW} эпизодов: {final_avg_ql:.3f}")
    print("Выученная политика (Q-Learning):")
    print_policy(ql_agent.q_table, n_states)

    print("\nОбучение SARSA")
    np.random.seed(SEED)
    sarsa_agent = SARSAAgent(n_states, n_actions)
    sarsa_rewards, sarsa_epsilons = train_sarsa(sarsa_agent, n_episodes=N_EPISODES, seed=SEED)

    final_avg_sarsa = np.mean(sarsa_rewards[-MOVING_AVG_WINDOW:])
    print(f"\nSARSA: средняя награда за последние {MOVING_AVG_WINDOW} эпизодов: {final_avg_sarsa:.3f}")
    print("Выученная политика (SARSA):")
    print_policy(sarsa_agent.q_table, n_states)

    # визуализация (п.4 задания): сходимость, epsilon, Q-таблица и политика
    # для каждого алгоритма отдельно, плюс совместные графики сравнения
    plot_convergence(ql_rewards, label="Q-Learning", color="blue", save_path=os.path.join(PLOTS_DIR, "ql_convergence.png"))
    plot_convergence(sarsa_rewards, label="SARSA", color="green", save_path=os.path.join(PLOTS_DIR, "sarsa_convergence.png"))

    plot_epsilon(ql_epsilons, label="Q-Learning", color="blue", save_path=os.path.join(PLOTS_DIR, "ql_epsilon.png"))
    plot_epsilon(sarsa_epsilons, label="SARSA", color="green", save_path=os.path.join(PLOTS_DIR, "sarsa_epsilon.png"))

    plot_q_heatmap(ql_agent.q_table, label="Q-Learning", save_path=os.path.join(PLOTS_DIR, "ql_heatmap.png"))
    plot_q_heatmap(sarsa_agent.q_table, label="SARSA", save_path=os.path.join(PLOTS_DIR, "sarsa_heatmap.png"))

    plot_policy(ql_agent.q_table, label="Q-Learning", save_path=os.path.join(PLOTS_DIR, "ql_policy.png"))
    plot_policy(sarsa_agent.q_table, label="SARSA", save_path=os.path.join(PLOTS_DIR, "sarsa_policy.png"))

    plot_comparison(ql_rewards, sarsa_rewards, save_path=os.path.join(PLOTS_DIR, "comparison.png"))
    plot_policy_diff(ql_agent.q_table, sarsa_agent.q_table, save_path=os.path.join(PLOTS_DIR, "policy_diff.png"))

    # демонстрация выученной политики в реальном времени (рендер среды)
    print("\n=== Демонстрация: Q-Learning ===")
    demo_episode(ql_agent, delay=0.5)

    print("\n=== Демонстрация: SARSA ===")
    demo_episode(sarsa_agent, delay=0.5)
