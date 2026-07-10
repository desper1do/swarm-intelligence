# visualize.py
import numpy as np
import matplotlib.pyplot as plt
from config import MOVING_AVG_WINDOW, MAP_NAME

ARROWS = {0: "←", 1: "↓", 2: "→", 3: "↑"}

# стандартная разметка FrozenLake 4x4 (для подсветки дыр/цели)
DEFAULT_4x4 = ["SFFF", "FHFH", "FFFH", "HFFG"]


def moving_average(data, window=MOVING_AVG_WINDOW):
    # скользящее среднее для сглаживания шумного графика наград
    data = np.array(data, dtype=float)
    if len(data) < window:
        return data
    kernel = np.ones(window) / window
    return np.convolve(data, kernel, mode="valid")


def plot_convergence(rewards_history, label="Agent", color="blue", save_path=None):
    # график сходимости - средняя награда по скользящему окну от номера эпизода (п.4.1 задания)
    smoothed = moving_average(rewards_history)
    plt.figure(figsize=(8, 5))
    plt.plot(smoothed, color=color, label=f"{label} (скользящее среднее, окно={MOVING_AVG_WINDOW})")
    plt.xlabel("Номер эпизода")
    plt.ylabel("Средняя награда")
    plt.title(f"Сходимость обучения: {label}")
    plt.legend()
    plt.grid(alpha=0.3)
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_epsilon(epsilon_history, label="Agent", color="blue", save_path=None):
    # график epsilon - изменение параметра исследования по эпизодам (п.4.1 задания)
    plt.figure(figsize=(8, 5))
    plt.plot(epsilon_history, color=color, label=label)
    plt.xlabel("Номер эпизода")
    plt.ylabel("Epsilon")
    plt.title(f"Уменьшение epsilon по эпизодам: {label}")
    plt.legend()
    plt.grid(alpha=0.3)
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_q_heatmap(q_table, label="Agent", save_path=None):
    # тепловая карта Q-значений: строки - состояния, столбцы - действия
    plt.figure(figsize=(5, 8))
    im = plt.imshow(q_table, cmap="viridis", aspect="auto")
    plt.colorbar(im, label="Q-значение")
    plt.xticks(range(4), [ARROWS[i] for i in range(4)])
    plt.yticks(range(q_table.shape[0]), range(q_table.shape[0]))
    plt.xlabel("Действие")
    plt.ylabel("Состояние")
    plt.title(f"Q-таблица: {label}")
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_policy(q_table, label="Agent", map_layout=None, save_path=None):
    # политика на сетке (стрелки поверх карты, дыры и цель подсвечены отдельно)
    if map_layout is None:
        map_layout = DEFAULT_4x4
    side = len(map_layout)

    fig, ax = plt.subplots(figsize=(5, 5))
    for row in range(side):
        for col in range(side):
            state = row * side + col
            cell = map_layout[row][col]

            if cell == "H":
                ax.add_patch(plt.Rectangle((col, side - 1 - row), 1, 1, color="lightcoral"))
                text = "H"
            elif cell == "G":
                ax.add_patch(plt.Rectangle((col, side - 1 - row), 1, 1, color="lightgreen"))
                text = "G"
            else:
                action = np.argmax(q_table[state])
                text = ARROWS[action]

            ax.text(col + 0.5, side - 1 - row + 0.5, text,
                     ha="center", va="center", fontsize=20)

    ax.set_xlim(0, side)
    ax.set_ylim(0, side)
    ax.set_xticks(range(side + 1))
    ax.set_yticks(range(side + 1))
    ax.grid(True)
    ax.set_title(f"Выученная политика: {label}")
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()