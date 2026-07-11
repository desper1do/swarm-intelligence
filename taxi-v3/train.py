from agent_sarsa import sarsa
from agent_qlearning import q_learning
from agent_reinforce import reinforce
from evaluate import evaluate_agent, evaluate_reinforce
from config import (EPISODES, ALPHA, GAMMA, EPS_START, EPS_END, EPS_DECAY,
                     REINFORCE_EPISODES, REINFORCE_LR, HIDDEN_DIM)


def train_all(train_env):
    print("Обучение SARSA")
    Q_sarsa, rewards_sarsa = sarsa(train_env, num_episodes=EPISODES, alpha=ALPHA, gamma=GAMMA,
                                    epsilon_start=EPS_START, epsilon_end=EPS_END, epsilon_decay=EPS_DECAY)
    avg_sarsa, _ = evaluate_agent(train_env, Q_sarsa)
    print(f"SARSA: средняя награда при жадной оценке = {avg_sarsa:.2f}")

    print("Обучение Q-Learning")
    Q_qlearning, rewards_qlearning = q_learning(train_env, num_episodes=EPISODES, alpha=ALPHA, gamma=GAMMA,
                                                 epsilon_start=EPS_START, epsilon_end=EPS_END, epsilon_decay=EPS_DECAY)
    avg_qlearning, _ = evaluate_agent(train_env, Q_qlearning)
    print(f"Q-Learning: средняя награда при жадной оценке = {avg_qlearning:.2f}")

    print("Обучение REINFORCE")
    policy_net, rewards_reinforce = reinforce(train_env, num_episodes=REINFORCE_EPISODES,
                                               gamma=GAMMA, lr=REINFORCE_LR, hidden_dim=HIDDEN_DIM)
    avg_reinforce, _ = evaluate_reinforce(train_env, policy_net)
    print(f"REINFORCE: средняя награда при жадной оценке = {avg_reinforce:.2f}")

    return {
        "sarsa": {"agent": Q_sarsa, "rewards": rewards_sarsa, "avg": avg_sarsa},
        "qlearning": {"agent": Q_qlearning, "rewards": rewards_qlearning, "avg": avg_qlearning},
        "reinforce": {"agent": policy_net, "rewards": rewards_reinforce, "avg": avg_reinforce},
    }