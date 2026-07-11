# Задание 1 (самостоятельная часть): результаты работы.
# Обучаем все три алгоритма, строим общий график сравнения и итоговую
# столбчатую диаграмму средних наград. Всё сохраняется в plots/.

import os
import pickle
import matplotlib
matplotlib.use("Agg")  # без окна, только сохранение в файл
import matplotlib.pyplot as plt
import numpy as np
import torch

import gymnasium as gym
from config import (SEED, EPISODES, ALPHA, GAMMA, EPS_START, EPS_END, EPS_DECAY,
                    REINFORCE_EPISODES, REINFORCE_LR, HIDDEN_DIM)
from agent_sarsa import sarsa
from agent_qlearning import q_learning
from agent_reinforce import reinforce
from evaluate import evaluate_agent, evaluate_reinforce

PLOTS = os.path.join(os.path.dirname(__file__), "plots")


def moving_average(data, w):
    return np.convolve(data, np.ones(w) / w, mode="valid")


def main():
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    env = gym.make("Taxi-v4")

    # обучаем три алгоритма с одинаковыми гиперпараметрами
    Q_sarsa, r_sarsa = sarsa(env, EPISODES, ALPHA, GAMMA, EPS_START, EPS_END, EPS_DECAY)
    avg_sarsa, _ = evaluate_agent(env, Q_sarsa)

    Q_ql, r_ql = q_learning(env, EPISODES, ALPHA, GAMMA, EPS_START, EPS_END, EPS_DECAY)
    avg_ql, _ = evaluate_agent(env, Q_ql)

    policy_net, r_reinf = reinforce(env, REINFORCE_EPISODES, GAMMA, REINFORCE_LR, HIDDEN_DIM)
    avg_reinf, _ = evaluate_reinforce(env, policy_net)
    env.close()

    print(f"\nИтог: SARSA={avg_sarsa:.2f} | Q-Learning={avg_ql:.2f} | REINFORCE={avg_reinf:.2f}")

    # график сравнения кривых обучения (скользящее среднее)
    w = 100
    plt.figure(figsize=(10, 6))
    plt.plot(moving_average(r_sarsa, w), label="SARSA")
    plt.plot(moving_average(r_ql, w), label="Q-Learning")
    plt.plot(moving_average(r_reinf, w), label="REINFORCE")
    plt.xlabel("Эпизод")
    plt.ylabel(f"Награда (скользящее среднее, окно={w})")
    plt.title("Сравнение алгоритмов на Taxi-v4")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS, "comparison_curves.png"), dpi=130)
    plt.close()

    # столбчатая диаграмма итоговых средних наград (жадная оценка)
    plt.figure(figsize=(7, 5))
    names = ["SARSA", "Q-Learning", "REINFORCE"]
    vals = [avg_sarsa, avg_ql, avg_reinf]
    bars = plt.bar(names, vals, color=["#4C72B0", "#55A868", "#C44E52"])
    plt.axhline(0, color="black", linewidth=0.8)
    plt.ylabel("Средняя награда (100 эпизодов, жадная политика)")
    plt.title("Итоговая производительность алгоритмов")
    for b, v in zip(bars, vals):
        plt.text(b.get_x() + b.get_width() / 2, v, f"{v:.1f}",
                 ha="center", va="bottom" if v >= 0 else "top")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS, "final_rewards_bar.png"), dpi=130)
    plt.close()

    # кэшируем Q-таблицу Q-Learning для задания 3 (lambda в defaultdict не пиклится,
    # поэтому сохраняем как обычный dict)
    with open(os.path.join(PLOTS, "q_table_ql.pkl"), "wb") as f:
        pickle.dump({s: np.asarray(a) for s, a in Q_ql.items()}, f)

    print("Графики сохранены в plots/: comparison_curves.png, final_rewards_bar.png")


if __name__ == "__main__":
    main()
