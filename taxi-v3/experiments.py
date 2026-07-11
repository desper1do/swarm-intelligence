# Задание 2 (самостоятельная часть): влияние гиперпараметров Q-Learning.
# Прогоняем Q-Learning с разными alpha и epsilon_decay, строим кривые обучения
# и определяем оптимальные значения по средней награде на последних эпизодах.

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import gymnasium as gym
from config import SEED, EPISODES, GAMMA, EPS_START, EPS_END
from agent_qlearning import q_learning
from evaluate import evaluate_agent

PLOTS = os.path.join(os.path.dirname(__file__), "plots")

ALPHAS = [0.01, 0.1, 0.5]              # скорость обучения
EPS_DECAYS = [0.99, 0.999, 0.9995]     # скорость затухания эпсилон


def moving_average(data, w=100):
    return np.convolve(data, np.ones(w) / w, mode="valid")


def run_qlearning(env, alpha, eps_decay):
    # один прогон с фиксированным сидом для честного сравнения
    np.random.seed(SEED)
    Q, rewards = q_learning(env, EPISODES, alpha, GAMMA, EPS_START, EPS_END, eps_decay)
    final = np.mean(rewards[-100:])      # средняя награда на хвосте обучения
    greedy, _ = evaluate_agent(env, Q)   # награда чистой жадной политики
    return rewards, final, greedy


def sweep(param_name, values, fixed_label, run_fn):
    # общий помощник: гоняем набор значений и рисуем кривые на одном графике
    plt.figure(figsize=(10, 6))
    best_val, best_score = None, -1e9
    print(f"\n=== Эксперимент: {param_name} (при {fixed_label}) ===")
    for v in values:
        rewards, final, greedy = run_fn(v)
        plt.plot(moving_average(rewards), label=f"{param_name}={v}")
        print(f"  {param_name}={v:<7} | хвост обучения={final:6.2f} | жадная оценка={greedy:6.2f}")
        if greedy > best_score:
            best_score, best_val = greedy, v
    print(f"  -> лучшее значение {param_name} = {best_val} (жадная оценка {best_score:.2f})")

    plt.xlabel("Эпизод")
    plt.ylabel("Награда (скользящее среднее, окно=100)")
    plt.title(f"Q-Learning: влияние {param_name}")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    out = os.path.join(PLOTS, f"exp_{param_name}.png")
    plt.savefig(out, dpi=130)
    plt.close()
    return best_val


def main():
    env = gym.make("Taxi-v4")

    # эксперимент 1: варьируем alpha, epsilon_decay фиксируем на 0.999
    best_alpha = sweep("alpha", ALPHAS, "epsilon_decay=0.999",
                       lambda a: run_qlearning(env, a, 0.999))

    # эксперимент 2: варьируем epsilon_decay, alpha фиксируем на 0.1
    best_decay = sweep("epsilon_decay", EPS_DECAYS, "alpha=0.1",
                       lambda d: run_qlearning(env, 0.1, d))

    env.close()
    print(f"\nОптимум: alpha={best_alpha}, epsilon_decay={best_decay}")
    print("Графики сохранены в plots/: exp_alpha.png, exp_epsilon_decay.png")


if __name__ == "__main__":
    main()
