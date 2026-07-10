# compare.py
import numpy as np
import matplotlib.pyplot as plt
from visualize import moving_average
from config import COLOR_QMIX, COLOR_MADDPG


def plot_comparison(qmix_rewards, maddpg_rewards, save_path=None):
    # сравнительный график сходимости QMIX vs MADDPG (п.4.3 задания).
    # у алгоритмов разное число эпизодов (MADDPG короче из-за нестабильности при
    # долгом обучении, см. config.py) - ось x у каждой линии своя, это ожидаемо
    plt.figure(figsize=(9, 5))
    plt.plot(moving_average(qmix_rewards), color=COLOR_QMIX, label="QMIX")
    plt.plot(moving_average(maddpg_rewards), color=COLOR_MADDPG, label="MADDPG")
    plt.xlabel("Номер эпизода")
    plt.ylabel("Суммарная награда (скользящее среднее)")
    plt.title("Сравнение сходимости: QMIX vs MADDPG")
    plt.legend()
    plt.grid(alpha=0.3)
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_success_rate_comparison(qmix_success, maddpg_success, save_path=None):
    # совместный success rate - дополняет график наград (который сильно
    # зашумлен штрафом за столкновения) честной картиной, решает ли алгоритм
    # саму задачу поиска пути, а не только штрафы избегает
    plt.figure(figsize=(9, 5))
    plt.plot(moving_average(qmix_success), color=COLOR_QMIX, label="QMIX")
    plt.plot(moving_average(maddpg_success), color=COLOR_MADDPG, label="MADDPG")
    plt.xlabel("Номер эпизода")
    plt.ylabel("Доля агентов, дошедших до цели (скользящее среднее)")
    plt.ylim(-0.05, 1.05)
    plt.title("Сравнение success rate: QMIX vs MADDPG")
    plt.legend()
    plt.grid(alpha=0.3)
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_time_memory(hist_qmix, hist_maddpg, save_path=None):
    # сравнение времени обучения и использования памяти (п.4.3 задания)
    labels = ["QMIX", "MADDPG"]
    times = [hist_qmix["train_time"], hist_maddpg["train_time"]]
    mem = [hist_qmix["peak_memory_mb"], hist_maddpg["peak_memory_mb"]]
    colors = [COLOR_QMIX, COLOR_MADDPG]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    ax1.bar(labels, times, color=colors)
    ax1.set_ylabel("Время обучения, с")
    ax1.set_title("Время обучения")
    ax1.grid(alpha=0.3, axis="y")

    ax2.bar(labels, mem, color=colors)
    ax2.set_ylabel("Пиковая память GPU, МБ")
    ax2.set_title("Использование памяти")
    ax2.grid(alpha=0.3, axis="y")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_collisions(qmix_collisions, maddpg_collisions, save_path=None):
    # визуализация столкновений - частота коллизий по эпизодам (п.4.3 задания)
    plt.figure(figsize=(9, 5))
    plt.plot(moving_average(qmix_collisions), color=COLOR_QMIX, label="QMIX")
    plt.plot(moving_average(maddpg_collisions), color=COLOR_MADDPG, label="MADDPG")
    plt.xlabel("Номер эпизода")
    plt.ylabel("Столкновений за эпизод (скользящее среднее)")
    plt.title("Частота столкновений по эпизодам")
    plt.legend()
    plt.grid(alpha=0.3)
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
