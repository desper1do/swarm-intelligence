# compare.py
import numpy as np
import matplotlib.pyplot as plt
from visualize import moving_average, ARROWS, get_grid_layout, _draw_grid_base


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


def plot_policy_diff(ql_q_table, sarsa_q_table, save_path=None):
    # сетка, где для каждой клетки показаны ОБЕ стрелки,
    # сверху политика Q-Learning (синим), снизу SARSA (зеленым).
    # различающиеся клетки подсвечены желтым фоном (п.4.2 задания)
    rows, cols, start, goal, cliff_cells = get_grid_layout()

    fig, ax = plt.subplots(figsize=(10, 4))
    _draw_grid_base(ax)

    for row in range(rows):
        for col in range(cols):
            if (row, col) in cliff_cells or (row, col) == goal:
                continue
            state = row * cols + col
            y = rows - 1 - row

            ql_action = np.argmax(ql_q_table[state])
            sarsa_action = np.argmax(sarsa_q_table[state])

            if ql_action != sarsa_action:
                ax.add_patch(plt.Rectangle((col, y), 1, 1, color="gold", alpha=0.4))

            ax.text(col + 0.5, y + 0.65, ARROWS[ql_action],
                     ha="center", va="center", fontsize=12, color="blue")
            ax.text(col + 0.5, y + 0.3, ARROWS[sarsa_action],
                     ha="center", va="center", fontsize=12, color="green")

    ax.set_title("Сравнение политик: синий=Q-Learning, зелёный=SARSA\n(желтый фон - расхождение)")
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
