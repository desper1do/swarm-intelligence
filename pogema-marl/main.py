# main.py
import os
import torch
import numpy as np

from config import QMIX, MADDPG, SEED
from animate import animate_qmix, animate_maddpg
from demo import rollout_qmix, rollout_maddpg
from visualize import (
    plot_convergence, plot_epsilon, plot_sigma, plot_q_tot, plot_losses,
    plot_agent_q_heatmap, plot_action_heatmap, plot_trajectory, plot_success_rate,
)
from compare import plot_comparison, plot_time_memory, plot_collisions, plot_success_rate_comparison

PLOTS_DIR = os.path.join(os.path.dirname(__file__), "plots")

if __name__ == "__main__":
    os.makedirs(PLOTS_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Устройство: {device}")

    print(f"\n=== Обучение QMIX ({QMIX.N_EPISODES} эпизодов, анимация - закройте окно по завершении) ===")
    qmix_agent, qmix_hist = animate_qmix(QMIX.N_EPISODES, device, SEED)
    print(f"QMIX: время={qmix_hist['train_time']:.1f}с, пик памяти={qmix_hist['peak_memory_mb']:.1f}МБ")

    print(f"\n=== Обучение MADDPG ({MADDPG.N_EPISODES} эпизодов, анимация - закройте окно по завершении) ===")
    maddpg_agent, maddpg_hist = animate_maddpg(MADDPG.N_EPISODES, device, SEED)
    print(f"MADDPG: время={maddpg_hist['train_time']:.1f}с, пик памяти={maddpg_hist['peak_memory_mb']:.1f}МБ")

    # визуализация QMIX (п.4.1 задания)
    plot_convergence(qmix_hist["reward"], "QMIX", "#2c5282", os.path.join(PLOTS_DIR, "qmix_convergence.png"))
    plot_success_rate(qmix_hist["success_rate"], "QMIX", "#2c5282", os.path.join(PLOTS_DIR, "qmix_success_rate.png"))
    plot_epsilon(qmix_hist["epsilon"], "QMIX", "#2c5282", os.path.join(PLOTS_DIR, "qmix_epsilon.png"))
    plot_q_tot(qmix_hist["q_tot"], "QMIX", "#2c5282", os.path.join(PLOTS_DIR, "qmix_q_tot.png"))

    qmix_rollout = rollout_qmix(qmix_agent, device, SEED)
    plot_agent_q_heatmap(qmix_rollout["q_values"], "QMIX", os.path.join(PLOTS_DIR, "qmix_q_heatmap.png"))
    plot_trajectory(qmix_rollout, "QMIX", os.path.join(PLOTS_DIR, "qmix_trajectory.png"))

    # визуализация MADDPG (п.4.2 задания)
    plot_convergence(maddpg_hist["reward"], "MADDPG", "#2d6a4f", os.path.join(PLOTS_DIR, "maddpg_convergence.png"))
    plot_success_rate(maddpg_hist["success_rate"], "MADDPG", "#2d6a4f", os.path.join(PLOTS_DIR, "maddpg_success_rate.png"))
    plot_losses(maddpg_hist["critic_loss"], maddpg_hist["actor_loss"], os.path.join(PLOTS_DIR, "maddpg_losses.png"))
    plot_sigma(maddpg_hist["sigma"], "MADDPG", "#2d6a4f", os.path.join(PLOTS_DIR, "maddpg_sigma.png"))

    maddpg_rollout = rollout_maddpg(maddpg_agent, device, SEED)
    plot_action_heatmap(maddpg_rollout["actions"], "MADDPG", os.path.join(PLOTS_DIR, "maddpg_action_heatmap.png"))
    plot_trajectory(maddpg_rollout, "MADDPG", os.path.join(PLOTS_DIR, "maddpg_trajectory.png"))

    # совместная визуализация (п.4.3 задания)
    plot_comparison(qmix_hist["reward"], maddpg_hist["reward"], os.path.join(PLOTS_DIR, "comparison.png"))
    plot_success_rate_comparison(
        qmix_hist["success_rate"], maddpg_hist["success_rate"], os.path.join(PLOTS_DIR, "success_rate_comparison.png")
    )
    plot_time_memory(qmix_hist, maddpg_hist, os.path.join(PLOTS_DIR, "time_memory.png"))
    plot_collisions(qmix_hist["collisions"], maddpg_hist["collisions"], os.path.join(PLOTS_DIR, "collisions.png"))

    print("\nГотово. Графики сохранены в", PLOTS_DIR)
