# visualize.py
import numpy as np
import matplotlib.pyplot as plt
from config import MOVING_AVG_WINDOW

ACTION_NAMES = ["Стоять", "Вверх", "Вниз", "Влево", "Вправо"]
AGENT_COLORS = ["#e63946", "#457b9d", "#f4a261", "#2a9d8f", "#9d4edd", "#ffb703", "#8338ec", "#06d6a0"]


def moving_average(data, window=MOVING_AVG_WINDOW):
    # скользящее среднее для сглаживания шумного графика наград, NaN (пропущенные
    # значения loss на шагах без обучения) из данных выбрасываем
    data = np.asarray(data, dtype=float)
    data = data[~np.isnan(data)] if np.isnan(data).any() else data
    if len(data) < window:
        return data
    kernel = np.ones(window) / window
    return np.convolve(data, kernel, mode="valid")


def plot_success_rate(success_rate_history, label, color, save_path=None):
    # доля агентов, дошедших до цели, по эпизодам - награда сильно зашумлена
    # штрафом за столкновения (см. config.COLLISION_PENALTY), поэтому success_rate
    # честнее показывает, решает ли алгоритм саму задачу поиска пути
    smoothed = moving_average(success_rate_history)
    plt.figure(figsize=(8, 5))
    plt.plot(smoothed, color=color, label=f"{label} (скользящее среднее, окно={MOVING_AVG_WINDOW})")
    plt.xlabel("Номер эпизода")
    plt.ylabel("Доля агентов, дошедших до цели")
    plt.ylim(-0.05, 1.05)
    plt.title(f"Success rate: {label}")
    plt.legend()
    plt.grid(alpha=0.3)
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_convergence(rewards_history, label, color, save_path=None):
    # график сходимости - средняя награда по скользящему окну от номера эпизода
    smoothed = moving_average(rewards_history)
    plt.figure(figsize=(8, 5))
    plt.plot(smoothed, color=color, label=f"{label} (скользящее среднее, окно={MOVING_AVG_WINDOW})")
    plt.xlabel("Номер эпизода")
    plt.ylabel("Суммарная награда")
    plt.title(f"Сходимость обучения: {label}")
    plt.legend()
    plt.grid(alpha=0.3)
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_epsilon(epsilon_history, label, color, save_path=None):
    # график epsilon (QMIX, п.4.1 задания)
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


def plot_sigma(sigma_history, label, color, save_path=None):
    # анализ шума (MADDPG, п.4.2 задания)
    plt.figure(figsize=(8, 5))
    plt.plot(sigma_history, color=color, label=label)
    plt.xlabel("Номер эпизода")
    plt.ylabel("Sigma (std гауссовского шума)")
    plt.title(f"Изменение exploration noise по эпизодам: {label}")
    plt.legend()
    plt.grid(alpha=0.3)
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_q_tot(q_tot_history, label, color, save_path=None):
    # график Q_tot (QMIX, п.4.1 задания)
    smoothed = moving_average(q_tot_history)
    plt.figure(figsize=(8, 5))
    plt.plot(smoothed, color=color, label=label)
    plt.xlabel("Номер эпизода")
    plt.ylabel("Среднее Q_tot")
    plt.title(f"Изменение совместного Q-значения: {label}")
    plt.legend()
    plt.grid(alpha=0.3)
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_losses(critic_loss_history, actor_loss_history, save_path=None):
    # функции потерь критика и актера (MADDPG, п.4.2 задания)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.plot(moving_average(critic_loss_history), color="#e63946")
    ax1.set_xlabel("Номер эпизода")
    ax1.set_ylabel("Critic loss (MSE)")
    ax1.set_title("Функция потерь критика")
    ax1.grid(alpha=0.3)

    ax2.plot(moving_average(actor_loss_history), color="#457b9d")
    ax2.set_xlabel("Номер эпизода")
    ax2.set_ylabel("Actor loss (-Q)")
    ax2.set_title("Функция потерь актера")
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_agent_q_heatmap(q_values, label, save_path=None):
    # тепловая карта Q-значений выбранного агента (QMIX, п.4.1 задания):
    # по одному эпизоду жадного прогона, строки - шаги эпизода, столбцы - действия
    plt.figure(figsize=(5, 8))
    im = plt.imshow(q_values, cmap="viridis", aspect="auto")
    plt.colorbar(im, label="Q-значение")
    plt.xticks(range(len(ACTION_NAMES)), ACTION_NAMES, rotation=45)
    plt.xlabel("Действие")
    plt.ylabel("Шаг эпизода")
    plt.title(f"Q-значения агента 0: {label}")
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_action_heatmap(actions, label, save_path=None):
    # тепловая карта выбранных действий - визуализация политики (MADDPG, п.4.2 задания):
    # строки - агенты, столбцы - шаги эпизода, цвет - номер выбранного действия
    plt.figure(figsize=(10, 3))
    im = plt.imshow(actions.T, cmap="tab10", aspect="auto", vmin=0, vmax=4)
    cbar = plt.colorbar(im, ticks=range(5))
    cbar.ax.set_yticklabels(ACTION_NAMES)
    plt.xlabel("Шаг эпизода")
    plt.ylabel("Агент")
    plt.yticks(range(actions.shape[1]))
    plt.title(f"Выбранные действия по шагам: {label}")
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_trajectory(rollout, label, save_path=None):
    # траектории агентов на карте POGEMA (п.4.1/4.2 задания): препятствия серые,
    # старт - квадрат, цель - звезда своего цвета, путь агента - линия того же цвета
    obstacles = rollout["obstacles"]
    positions = np.array(rollout["positions"])  # (T+1, n_agents, 2) - (row, col)
    targets = rollout["targets"]
    n_agents = positions.shape[1]
    rows, cols = obstacles.shape

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(obstacles, cmap="Greys", origin="upper", alpha=0.6)

    for i in range(n_agents):
        color = AGENT_COLORS[i % len(AGENT_COLORS)]
        path = positions[:, i, :]
        ax.plot(path[:, 1], path[:, 0], color=color, linewidth=2, marker="o", markersize=3, label=f"агент {i}")
        ax.plot(path[0, 1], path[0, 0], color=color, marker="s", markersize=12, markeredgecolor="black")
        tx, ty = targets[i]
        ax.plot(ty, tx, color=color, marker="*", markersize=16, markeredgecolor="black")

    ax.set_xlim(-0.5, cols - 0.5)
    ax.set_ylim(rows - 0.5, -0.5)
    ax.set_title(f"Траектории агентов: {label}\n(квадрат - старт, звезда - цель)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.05), ncol=min(n_agents, 4))
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
