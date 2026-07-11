# Задание 3 (самостоятельная часть): анализ оптимальной политики Q-Learning.
# Выводим наиболее предпочтительное действие для каждого состояния и
# визуализируем политику на карте 5x5 для разных сценариев (где пассажир,
# куда везём). Разбираем поведение агента по зонам.

import os
import pickle
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import gymnasium as gym
from config import SEED, EPISODES, ALPHA, GAMMA, EPS_START, EPS_END, EPS_DECAY
from agent_qlearning import q_learning

PLOTS = os.path.join(os.path.dirname(__file__), "plots")

# буквенно-символьное обозначение 6 действий Taxi
ACTION_SYM = {0: "↓", 1: "↑", 2: "→", 3: "←", 4: "P", 5: "D"}
ACTION_NAME = {0: "Юг", 1: "Север", 2: "Восток", 3: "Запад", 4: "Забрать", 5: "Высадить"}

# 4 фиксированные локации: R, G, Y, B (индекс -> (row, col))
LOCS = {0: (0, 0), 1: (0, 4), 2: (4, 0), 3: (4, 3)}
LOC_NAME = {0: "R", 1: "G", 2: "Y", 3: "B"}

# вертикальные стены карты Taxi: (row, col) означает стену между col и col+1
WALLS = {(0, 1), (1, 1), (3, 0), (3, 2), (4, 0), (4, 2)}


def load_or_train_Q():
    # берём кэш из task1_comparison, иначе обучаем заново
    cache = os.path.join(PLOTS, "q_table_ql.pkl")
    if os.path.exists(cache):
        with open(cache, "rb") as f:
            return pickle.load(f)
    print("Кэш q_table_ql.pkl не найден, обучаю Q-Learning заново...")
    np.random.seed(SEED)
    env = gym.make("Taxi-v4")
    Q, _ = q_learning(env, EPISODES, ALPHA, GAMMA, EPS_START, EPS_END, EPS_DECAY)
    env.close()
    return {s: np.asarray(a) for s, a in Q.items()}


def best_action(Q, state):
    q = Q.get(state)
    if q is None:
        return None                # состояние не встречалось при обучении
    return int(np.argmax(q))


def draw_scene(ax, env, Q, pass_idx, dest_idx, title):
    # рисуем карту 5x5 и лучшее действие в каждой клетке (для фиксированных
    # положения пассажира и пункта назначения)
    ax.set_xlim(0, 5)
    ax.set_ylim(0, 5)
    ax.invert_yaxis()              # строка 0 сверху, как в игре
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=10)

    # сетка
    for i in range(6):
        ax.axhline(i, color="lightgray", linewidth=0.8)
        ax.axvline(i, color="lightgray", linewidth=0.8)
    # внешняя рамка и стены
    for s in ("top", "bottom", "left", "right"):
        ax.spines[s].set_visible(False)
    ax.add_patch(plt.Rectangle((0, 0), 5, 5, fill=False, edgecolor="black", linewidth=2))
    for (r, c) in WALLS:
        ax.plot([c + 1, c + 1], [r, r + 1], color="black", linewidth=3)

    # подписи локаций R/G/Y/B
    for idx, (r, c) in LOCS.items():
        ax.text(c + 0.08, r + 0.28, LOC_NAME[idx], color="gray", fontsize=9, fontweight="bold")

    # лучшее действие в каждой клетке-положении такси
    for tr in range(5):
        for tc in range(5):
            state = env.unwrapped.encode(tr, tc, pass_idx, dest_idx)
            a = best_action(Q, state)
            sym = ACTION_SYM.get(a, "?")
            color = "black"
            if a == 4:
                color = "#1f77b4"   # забрать - синий
            elif a == 5:
                color = "#2ca02c"   # высадить - зелёный
            ax.text(tc + 0.5, tr + 0.6, sym, ha="center", va="center",
                    fontsize=16, color=color, fontweight="bold")

    # отмечаем цель поездки красной рамкой
    dr, dc = LOCS[dest_idx]
    ax.add_patch(plt.Rectangle((dc, dr), 1, 1, fill=False, edgecolor="red", linewidth=2))
    # если пассажир ждёт на локации (не в такси) - синяя рамка
    if pass_idx != 4:
        pr, pc = LOCS[pass_idx]
        ax.add_patch(plt.Rectangle((pc, pr), 1, 1, fill=False, edgecolor="blue",
                                   linewidth=2, linestyle="--"))


def print_action_table(Q):
    # текстовый вывод: распределение предпочтительных действий по всем состояниям
    counts = {a: 0 for a in ACTION_SYM}
    unknown = 0
    for s in range(500):
        a = best_action(Q, s)
        if a is None:
            unknown += 1
        else:
            counts[a] += 1
    print("\nРаспределение предпочтительных действий по 500 состояниям:")
    for a, n in counts.items():
        print(f"  {a} {ACTION_NAME[a]:<10}: {n:3d} состояний")
    print(f"  не встречалось при обучении: {unknown}")


def main():
    Q = load_or_train_Q()
    env = gym.make("Taxi-v4")

    print_action_table(Q)

    # два показательных сценария: пассажир ждёт vs пассажир уже в такси
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    draw_scene(axes[0], env, Q, pass_idx=0, dest_idx=1,
               title="Пассажир ждёт на R (синий),\nвезём в G (красный)")
    draw_scene(axes[1], env, Q, pass_idx=4, dest_idx=3,
               title="Пассажир в такси,\nвысадка в B (красный)")
    fig.suptitle("Оптимальная политика Q-Learning (лучшее действие в каждой клетке)", fontsize=12)
    plt.tight_layout()
    out = os.path.join(PLOTS, "policy_map.png")
    plt.savefig(out, dpi=130)
    plt.close()

    # проверяем ключевые клетки: забирает ли на R и высаживает ли на B
    s_pick = env.unwrapped.encode(0, 0, 0, 1)   # такси на R, пассажир на R
    s_drop = env.unwrapped.encode(4, 3, 4, 3)   # такси на B, пассажир в такси, цель B
    print(f"\nНа клетке R с ждущим пассажиром действие: {ACTION_NAME[best_action(Q, s_pick)]}")
    print(f"На клетке B с пассажиром в такси действие: {ACTION_NAME[best_action(Q, s_drop)]}")

    env.close()
    print("График сохранён в plots/: policy_map.png")


if __name__ == "__main__":
    main()
