# visualize.py
import numpy as np
import matplotlib.pyplot as plt
from config import MOVING_AVG_WINDOW, GRID_SHAPE, SEED
from environment import make_env

# в CliffWalking действия закодированы иначе, чем во FrozenLake: 0=вверх, 1=вправо,
# 2=вниз, 3=влево (см. gymnasium.envs.toy_text.cliffwalking)
ARROWS = {0: "↑", 1: "→", 2: "↓", 3: "←"}


def get_grid_layout():
    # обрыв (cliff) вся нижняя строка кроме первой и последней клетки,
    # старт - левый нижний угол, цель - правый нижний угол (как задано в gymnasium)
    rows, cols = GRID_SHAPE
    start = (rows - 1, 0)
    goal = (rows - 1, cols - 1)
    cliff_cells = {(rows - 1, col) for col in range(1, cols - 1)}
    return rows, cols, start, goal, cliff_cells


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
    # тепловая карта Q-значений: строки - состояния (48), столбцы - действия
    plt.figure(figsize=(5, 10))
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


def _draw_grid_base(ax):
    # общая отрисовка сетки 4x12: обрыв (красный), старт и цель (зеленый).
    # используется и в plot_policy, и в plot_trajectory, чтобы не дублировать код
    rows, cols, start, goal, cliff_cells = get_grid_layout()

    for row in range(rows):
        for col in range(cols):
            y = rows - 1 - row
            if (row, col) in cliff_cells:
                ax.add_patch(plt.Rectangle((col, y), 1, 1, color="lightcoral"))
                ax.text(col + 0.5, y + 0.5, "C", ha="center", va="center", fontsize=14)
            elif (row, col) == start:
                ax.add_patch(plt.Rectangle((col, y), 1, 1, color="khaki", alpha=0.6))
                ax.text(col + 0.2, y + 0.8, "S", ha="center", va="center", fontsize=10)
            elif (row, col) == goal:
                ax.add_patch(plt.Rectangle((col, y), 1, 1, color="lightgreen"))
                ax.text(col + 0.2, y + 0.8, "G", ha="center", va="center", fontsize=10)

    ax.set_xlim(0, cols)
    ax.set_ylim(0, rows)
    ax.set_xticks(range(cols + 1))
    ax.set_yticks(range(rows + 1))
    ax.grid(True)


def plot_policy(q_table, label="Agent", save_path=None):
    # политика на сетке (стрелки поверх карты, обрыв/старт/цель подсвечены отдельно, п.4.1 задания)
    rows, cols, start, goal, cliff_cells = get_grid_layout()
    fig, ax = plt.subplots(figsize=(10, 4))
    _draw_grid_base(ax)

    for row in range(rows):
        for col in range(cols):
            if (row, col) in cliff_cells or (row, col) == goal:
                continue
            state = row * cols + col
            action = np.argmax(q_table[state])
            y = rows - 1 - row
            ax.text(col + 0.5, y + 0.5, ARROWS[action], ha="center", va="center", fontsize=18)

    ax.set_title(f"Выученная политика: {label}")
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_trajectory(q_table, label="Agent", color="blue", max_steps=200, save_path=None):
    # траектория агента - жадный прогон по выученной Q-таблице, путь рисуется
    # поверх сетки (п.4.1 задания). Среда детерминирована, так что путь всегда
    # один и тот же для данной q_table
    rows, cols, start, goal, cliff_cells = get_grid_layout()
    env = make_env()
    state, info = env.reset(seed=SEED)

    positions = [np.unravel_index(state, GRID_SHAPE)]
    done = False
    steps = 0
    total_reward = 0
    while not done and steps < max_steps:
        action = np.argmax(q_table[state])
        state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward
        positions.append(np.unravel_index(state, GRID_SHAPE))
        steps += 1
    env.close()

    fig, ax = plt.subplots(figsize=(10, 4))
    _draw_grid_base(ax)

    xs = [col + 0.5 for row, col in positions]
    ys = [rows - 1 - row + 0.5 for row, col in positions]
    ax.plot(xs, ys, color=color, marker="o", markersize=6, linewidth=2, label=label)
    ax.plot(xs[0], ys[0], marker="s", markersize=12, color="black", label="старт")
    ax.plot(xs[-1], ys[-1], marker="*", markersize=16, color="gold", label="конец")

    ax.set_title(f"Траектория агента: {label} ({steps} шагов, награда {total_reward:.0f})")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.05), ncol=3)
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
