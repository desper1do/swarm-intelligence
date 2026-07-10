# compare.py
import numpy as np
import matplotlib.pyplot as plt
from visualize import moving_average, ARROWS, DEFAULT_4x4


def plot_comparison(ql_rewards, sarsa_rewards, save_path=None):
    # совместный график сходимости QL и SARSA (п.4.2 задания)
    ql_smooth = moving_average(ql_rewards)
    sarsa_smooth = moving_average(sarsa_rewards)

    plt.figure(figsize=(9, 5))
    plt.plot(ql_smooth, color="blue", label="Q-Learning")
    plt.plot(sarsa_smooth, color="green", label="SARSA")
    plt.xlabel("Номер эпизода")
    plt.ylabel("Средняя награда (скользящее окно)")
    plt.title("Сравнение сходимости: Q-Learning vs SARSA")
    plt.legend()
    plt.grid(alpha=0.3)
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_policy_diff(ql_q_table, sarsa_q_table, map_layout=None, save_path=None):
    # сетка, где для каждой клетки показаны ОБЕ стрелки,
    # сверху политика Q-Learning (синим), снизу SARSA (зеленым).
    # Различающиеся клетки подсвечены жёлтым фоном
    if map_layout is None:
        map_layout = DEFAULT_4x4
    side = len(map_layout)

    fig, ax = plt.subplots(figsize=(6, 6))
    for row in range(side):
        for col in range(side):
            state = row * side + col
            cell = map_layout[row][col]
            y = side - 1 - row

            if cell == "H":
                ax.add_patch(plt.Rectangle((col, y), 1, 1, color="lightcoral"))
                ax.text(col + 0.5, y + 0.5, "H", ha="center", va="center", fontsize=16)
                continue
            if cell == "G":
                ax.add_patch(plt.Rectangle((col, y), 1, 1, color="lightgreen"))
                ax.text(col + 0.5, y + 0.5, "G", ha="center", va="center", fontsize=16)
                continue

            ql_action = np.argmax(ql_q_table[state])
            sarsa_action = np.argmax(sarsa_q_table[state])

            if ql_action != sarsa_action:
                ax.add_patch(plt.Rectangle((col, y), 1, 1, color="khaki", alpha=0.5))

            ax.text(col + 0.5, y + 0.65, ARROWS[ql_action],
                     ha="center", va="center", fontsize=14, color="blue")
            ax.text(col + 0.5, y + 0.3, ARROWS[sarsa_action],
                     ha="center", va="center", fontsize=14, color="green")

    ax.set_xlim(0, side)
    ax.set_ylim(0, side)
    ax.set_xticks(range(side + 1))
    ax.set_yticks(range(side + 1))
    ax.grid(True)
    ax.set_title("Сравнение политик: синий=Q-Learning, зелёный=SARSA\n(желтый фон - расхождение)")
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()