"""
Визуализация результатов обучения MARL алгоритмов.

Строит графики из JSON логов, созданных во время обучения.

Примеры:
    # Один алгоритм
    python plot_results.py --log ./logs/maddpg/maddpg_10gen_terran_metrics.json

    # Сравнение нескольких алгоритмов
    python plot_results.py \
        --log ./logs/maddpg/maddpg_10gen_terran_metrics.json \
        --log ./logs/qmix/qmix_10gen_terran_metrics.json \
        --log ./logs/mappo/mappo_10gen_terran_metrics.json \
        --labels MADDPG QMIX MAPPO

    # Скользящее среднее
    python plot_results.py --log ./logs/maddpg/*.json --window 100

    # Сохранить без показа
    python plot_results.py --log ./logs/qmix/*.json --save_dir ./plots
"""

import os
import sys
import json
import argparse
import glob
from typing import List, Dict, Optional

import numpy as np


def load_metrics(path: str) -> List[Dict]:
    """Загружает метрики из JSON файла."""
    with open(path, "r") as f:
        return json.load(f)


def moving_average(data: np.ndarray, window: int) -> np.ndarray:
    """Вычисляет скользящее среднее."""
    if window <= 1:
        return data
    pad = window // 2
    cumsum = np.cumsum(np.pad(data, (pad, pad), mode="edge"))
    return (cumsum[window:] - cumsum[:-window]) / window


def extract_series(metrics: List[Dict], key: str) -> tuple:
    """Извлекает серию данных по ключу."""
    episodes = []
    values = []
    for m in metrics:
        if key in m:
            episodes.append(m.get("episode", len(episodes) + 1))
            values.append(m[key])
    return np.array(episodes), np.array(values, dtype=np.float32)


def plot_training_curves(
    log_paths: List[str],
    labels: Optional[List[str]] = None,
    window: int = 50,
    save_dir: Optional[str] = None,
    show: bool = True,
):
    """Строит графики обучения."""
    try:
        import matplotlib

        if not show:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("ERROR: matplotlib не установлен. Установите: pip install matplotlib")
        sys.exit(1)

    # Раскрываем glob patterns
    expanded_paths = []
    for p in log_paths:
        matches = glob.glob(p)
        expanded_paths.extend(matches if matches else [p])
    log_paths = expanded_paths

    if not log_paths:
        print("ERROR: Не найдены лог-файлы.")
        sys.exit(1)

    if labels is None:
        labels = [os.path.splitext(os.path.basename(p))[0] for p in log_paths]
    labels = labels[: len(log_paths)]
    while len(labels) < len(log_paths):
        labels.append(os.path.splitext(os.path.basename(log_paths[len(labels)]))[0])

    # Цвета
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]

    # Загружаем все метрики
    all_metrics = []
    for path in log_paths:
        if not os.path.exists(path):
            print(f"WARNING: Файл не найден: {path}")
            continue
        metrics = load_metrics(path)
        all_metrics.append(metrics)
        print(f"Loaded: {path} ({len(metrics)} episodes)")

    if not all_metrics:
        print("ERROR: Нет данных для построения графиков.")
        sys.exit(1)

    # Создаём фигуры
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("MARL Training Results", fontsize=16, fontweight="bold")

    plot_configs = [
        ("reward", "Episode Reward", axes[0, 0]),
        ("mean_reward_100", "Mean Reward (100 ep)", axes[0, 1]),
        ("win_rate_100", "Win Rate (100 ep)", axes[0, 2]),
        ("length", "Episode Length", axes[1, 0]),
        ("actor_loss", "Actor Loss", axes[1, 1]),
        ("critic_loss", "Critic Loss", axes[1, 2]),
    ]

    for key, title, ax in plot_configs:
        for idx, metrics in enumerate(all_metrics):
            episodes, values = extract_series(metrics, key)
            if len(episodes) == 0:
                continue

            label = labels[idx]
            color = colors[idx % len(colors)]

            # Сырая линия (прозрачная)
            ax.plot(episodes, values, alpha=0.2, color=color)

            # Скользящее среднее
            if len(values) >= window:
                smoothed = moving_average(values, window)
                # Выравниваем длины
                start = (len(values) - len(smoothed)) // 2
                ax.plot(
                    episodes[start : start + len(smoothed)],
                    smoothed,
                    label=label,
                    color=color,
                    linewidth=2,
                )
            else:
                ax.plot(episodes, values, label=label, color=color, linewidth=2)

        ax.set_title(title, fontsize=12)
        ax.set_xlabel("Episode")
        ax.set_ylabel(title)
        ax.grid(True, alpha=0.3)
        if len(all_metrics) > 1:
            ax.legend(loc="best", fontsize=9)

    plt.tight_layout()

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, "training_curves.png")
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")

    if show:
        plt.show()

    # === Дополнительные графики ===
    fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))
    fig2.suptitle("Extended Metrics", fontsize=14, fontweight="bold")

    # Epsilon / Noise
    ax = axes2[0]
    for idx, metrics in enumerate(all_metrics):
        for key in ["epsilon", "noise_scale"]:
            episodes, values = extract_series(metrics, key)
            if len(episodes) > 0:
                label = f"{labels[idx]} - {key}"
                color = colors[idx % len(colors)]
                ax.plot(episodes, values, label=label, color=color, linewidth=2)
    ax.set_title("Exploration Rate", fontsize=12)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Value")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9)

    # QMIX loss (если есть)
    ax = axes2[1]
    for idx, metrics in enumerate(all_metrics):
        episodes, values = extract_series(metrics, "qmix_loss")
        if len(episodes) > 0:
            label = labels[idx]
            color = colors[idx % len(colors)]
            if len(values) >= window:
                smoothed = moving_average(values, window)
                start = (len(values) - len(smoothed)) // 2
                ax.plot(episodes[start : start + len(smoothed)], smoothed, label=label, color=color, linewidth=2)
            else:
                ax.plot(episodes, values, label=label, color=color, linewidth=2)
    ax.set_title("QMIX Loss", fontsize=12)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Loss")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9)

    plt.tight_layout()

    if save_dir:
        save_path = os.path.join(save_dir, "extended_metrics.png")
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")

    if show:
        plt.show()

    print("Done!")


def main():
    parser = argparse.ArgumentParser(description="Plot MARL training results")
    parser.add_argument("--log", type=str, nargs="+", required=True,
                        help="Path(s) to metrics JSON files (supports glob)")
    parser.add_argument("--labels", type=str, nargs="+", default=None,
                        help="Labels for each log file")
    parser.add_argument("--window", type=int, default=50,
                        help="Moving average window size")
    parser.add_argument("--save_dir", type=str, default=None,
                        help="Directory to save plots (if set, plots are saved)")
    parser.add_argument("--no_show", action="store_true",
                        help="Do not show plots, only save")
    args = parser.parse_args()

    plot_training_curves(
        log_paths=args.log,
        labels=args.labels,
        window=args.window,
        save_dir=args.save_dir,
        show=not args.no_show,
    )


if __name__ == "__main__":
    main()
