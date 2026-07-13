"""
Утилиты для запуска и логирования экспериментов.
"""

import os
import time
import json
import numpy as np
import torch
from typing import Dict, List, Optional
from collections import deque


class Logger:
    """Простой логгер для отслеживания метрик обучения."""

    def __init__(self, log_dir: str, experiment_name: str):
        self.log_dir = log_dir
        self.experiment_name = experiment_name
        self.log_file = os.path.join(log_dir, f"{experiment_name}_log.txt")
        self.metrics_file = os.path.join(log_dir, f"{experiment_name}_metrics.json")

        os.makedirs(log_dir, exist_ok=True)

        self.metrics_history = []
        self.episode_rewards = deque(maxlen=100)
        self.episode_lengths = deque(maxlen=100)
        self.episode_win_rates = deque(maxlist=100) if hasattr(deque, "maxlist") else deque(maxlen=100)

        self.start_time = time.time()

    def log(self, message: str):
        """Записывает сообщение в лог."""
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{timestamp}] {message}"
        print(line)
        with open(self.log_file, "a") as f:
            f.write(line + "\n")

    def log_episode(
        self,
        episode: int,
        reward: float,
        length: int,
        won: bool = False,
        extra_info: Optional[Dict] = None,
    ):
        """Логирует завершение эпизода."""
        self.episode_rewards.append(reward)
        self.episode_lengths.append(length)
        self.episode_win_rates.append(1.0 if won else 0.0)

        mean_reward = np.mean(self.episode_rewards)
        mean_length = np.mean(self.episode_lengths)
        win_rate = np.mean(self.episode_win_rates)

        elapsed = time.time() - self.start_time

        metrics = {
            "episode": episode,
            "reward": reward,
            "mean_reward_100": mean_reward,
            "length": length,
            "mean_length_100": mean_length,
            "win": won,
            "win_rate_100": win_rate,
            "elapsed_time": elapsed,
        }

        if extra_info:
            metrics.update(extra_info)

        self.metrics_history.append(metrics)

        # Сохраняем метрики
        with open(self.metrics_file, "w") as f:
            json.dump(self.metrics_history, f, indent=2)

        self.log(
            f"Episode {episode} | Reward: {reward:.2f} | Mean(100): {mean_reward:.2f} | "
            f"Length: {length} | Win: {won} | WinRate(100): {win_rate:.2%} | "
            f"Time: {elapsed:.1f}s"
        )

        return metrics

    def log_training_step(self, step: int, losses: Dict[str, float]):
        """Логирует шаг обучения."""
        loss_str = " | ".join([f"{k}: {v:.4f}" for k, v in losses.items()])
        self.log(f"Step {step} | {loss_str}")


class Timer:
    """Контекстный менеджер для измерения времени выполнения."""

    def __init__(self, name: str = "Operation"):
        self.name = name
        self.start = None
        self.elapsed = None

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.elapsed = time.time() - self.start
        print(f"{self.name} took {self.elapsed:.2f}s")


def save_checkpoint(
    save_dir: str,
    algorithm_name: str,
    episode: int,
    models: Dict[str, torch.nn.Module],
    optimizers: Optional[Dict[str, torch.optim.Optimizer]] = None,
    extra_data: Optional[Dict] = None,
):
    """Сохраняет чекпоинт модели."""
    os.makedirs(save_dir, exist_ok=True)

    checkpoint = {
        "algorithm": algorithm_name,
        "episode": episode,
        "models": {k: v.state_dict() for k, v in models.items()},
    }

    if optimizers:
        checkpoint["optimizers"] = {k: v.state_dict() for k, v in optimizers.items()}

    if extra_data:
        checkpoint["extra"] = extra_data

    path = os.path.join(save_dir, f"{algorithm_name}_episode_{episode}.pt")
    torch.save(checkpoint, path)
    print(f"Checkpoint saved: {path}")
    return path


def load_checkpoint(
    path: str,
    models: Dict[str, torch.nn.Module],
    optimizers: Optional[Dict[str, torch.optim.Optimizer]] = None,
    device: str = "cuda",
) -> Dict:
    """Загружает чекпоинт модели."""
    checkpoint = torch.load(path, map_location=device)

    for k, v in checkpoint["models"].items():
        if k in models:
            models[k].load_state_dict(v)

    if optimizers and "optimizers" in checkpoint:
        for k, v in checkpoint["optimizers"].items():
            if k in optimizers:
                optimizers[k].load_state_dict(v)

    print(f"Checkpoint loaded: {path}")
    return checkpoint.get("extra", {})


def set_seed(seed: int, deterministic: bool = False):
    """Устанавливает seed для воспроизводимости."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True  # Ускорение при фиксированных размерах входов


def get_device() -> str:
    """Возвращает доступное устройство с диагностикой."""
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        print(f"GPU detected: {device_name}")
        print(f"CUDA version: {torch.version.cuda}")
        print(f"PyTorch CUDA: {torch.cuda.is_available()}")
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        return "cuda"
    else:
        print("WARNING: CUDA not available. Training on CPU.")
        print(f"PyTorch version: {torch.__version__}")
        print("To use GPU, install PyTorch with CUDA support:")
        print("  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121")
        return "cpu"
