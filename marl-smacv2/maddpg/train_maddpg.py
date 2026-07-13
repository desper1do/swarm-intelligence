"""
Скрипт обучения MADDPG в SMACv2.

Пример запуска:
    cd maddpg
    python train_maddpg.py --map_name 10gen_terran --n_episodes 5000 --batch_size 256
"""

import os
import sys
import argparse
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from common.starcraft_env import SMACv2Wrapper
from common.runner import Logger, set_seed, get_device
from maddpg.maddpg_agent import MADDPGTrainer


def train(args):
    """Основной цикл обучения MADDPG."""
    set_seed(args.seed, deterministic=False)
    device = get_device()

    # Создаём среду
    env = SMACv2Wrapper(
        map_name=args.map_name,
        debug=args.debug,
        conic_fov=args.conic_fov,
        use_unit_ranges=args.use_unit_ranges,
        headless=not args.render,
        step_mul=args.step_mul,
        seed=args.seed,
    )

    env_info = env.get_env_info()
    print(f"Environment Info: {env_info}")

    # Создаём тренер
    trainer = MADDPGTrainer(
        n_agents=env_info["n_agents"],
        obs_dim=env_info["obs_dim"],
        n_actions=env_info["n_actions"],
        state_dim=env_info["state_dim"],
        hidden_dims=args.hidden_dims,
        buffer_capacity=args.buffer_capacity,
        batch_size=args.batch_size,
        lr_actor=args.lr_actor,
        lr_critic=args.lr_critic,
        gamma=args.gamma,
        tau=args.tau,
        noise_scale=args.noise_scale,
        noise_decay=args.noise_decay,
        warmup_steps=args.warmup_steps,
        device=device,
    )

    # Логгер
    logger = Logger(args.log_dir, f"maddpg_{args.map_name}")
    logger.log(f"Starting MADDPG training on {args.map_name}")
    logger.log(f"Device: {device}")
    logger.log(f"Args: {args}")

    try:
        # Основной цикл обучения
        for episode in range(1, args.n_episodes + 1):
            observations, state = env.reset()
            avail_actions = env.get_avail_actions()

            episode_reward = 0
            episode_losses = {}
            train_steps = 0
            won = False

            for step in range(env.episode_limit):
                # Выбираем действия
                actions = trainer.select_actions(observations, avail_actions, explore=True)

                # Выполняем шаг
                next_observations, next_state, reward, terminated, info = env.step(actions)
                next_avail_actions = env.get_avail_actions()

                # Сохраняем переход
                trainer.store_transition(
                    state, observations, actions, reward,
                    next_state, next_observations, terminated,
                )

                episode_reward += reward
                observations = next_observations
                state = next_state
                avail_actions = next_avail_actions

                # Обучаем
                if len(trainer.replay_buffer) >= args.batch_size and step % args.train_interval == 0:
                    losses = trainer.train()
                    if losses:
                        for k, v in losses.items():
                            episode_losses[k] = episode_losses.get(k, 0) + v
                        train_steps += 1

                if terminated:
                    won = info.get("battle_won", False)
                    break

            trainer.episode_count = episode

            # Логируем
            avg_losses = {k: v / max(train_steps, 1) for k, v in episode_losses.items()}
            avg_losses["warmup"] = trainer.train_step < trainer.warmup_steps
            logger.log_episode(
                episode,
                episode_reward,
                step + 1,
                won=won,
                extra_info=avg_losses,
            )

            # Сохраняем чекпоинт
            if episode % args.save_interval == 0:
                os.makedirs(args.save_dir, exist_ok=True)
                save_path = os.path.join(args.save_dir, f"maddpg_{args.map_name}_ep{episode}.pt")
                trainer.save(save_path)

            # Оценка
            if episode % args.eval_interval == 0:
                eval_reward, eval_win = evaluate(trainer, env, args.eval_episodes, device)
                logger.log(f"Eval after {episode} episodes | Reward: {eval_reward:.2f} | WinRate: {eval_win:.2%}")

    except KeyboardInterrupt:
        logger.log("Training interrupted by user.")
        # Сохраняем чекпоинт при прерывании
        os.makedirs(args.save_dir, exist_ok=True)
        save_path = os.path.join(args.save_dir, f"maddpg_{args.map_name}_interrupted.pt")
        trainer.save(save_path)
        logger.log(f"Checkpoint saved: {save_path}")

    finally:
        env.close()
        logger.log("Environment closed. Exiting.")


def evaluate(trainer, env, n_episodes, device):
    """Оценивает политику без исследования."""
    total_reward = 0
    wins = 0

    for _ in range(n_episodes):
        observations, state = env.reset()
        avail_actions = env.get_avail_actions()
        episode_reward = 0

        for step in range(env.episode_limit):
            actions = trainer.select_actions(observations, avail_actions, explore=False)
            observations, state, reward, terminated, info = env.step(actions)
            avail_actions = env.get_avail_actions()
            episode_reward += reward

            if terminated:
                if info.get("battle_won", False):
                    wins += 1
                break

        total_reward += episode_reward

    return total_reward / n_episodes, wins / n_episodes


def main():
    parser = argparse.ArgumentParser(description="Train MADDPG on SMACv2")

    # Environment
    parser.add_argument("--map_name", type=str, default="10gen_terran",
                        help="SMACv2 map name (e.g., 10gen_terran, 3m, 8m, MMM)")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--conic_fov", action="store_true", help="Use conic field of view")
    parser.add_argument("--use_unit_ranges", action="store_true", default=True,
                        help="Use unit ranges")
    parser.add_argument("--render", action="store_true", default=False,
                        help="Enable StarCraft II visual rendering (slower)")
    parser.add_argument("--step_mul", type=int, default=8,
                        help="Game steps per agent step. Higher = faster but coarser. Try 16 or 32.")

    # Training
    parser.add_argument("--n_episodes", type=int, default=5000,
                        help="Number of training episodes")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size")
    parser.add_argument("--train_interval", type=int, default=1,
                        help="Train every N steps")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")

    # Model
    parser.add_argument("--hidden_dims", type=int, nargs="+", default=[256, 128],
                        help="Hidden layer dimensions")
    parser.add_argument("--lr_actor", type=float, default=1e-4,
                        help="Actor learning rate")
    parser.add_argument("--lr_critic", type=float, default=1e-3,
                        help="Critic learning rate")
    parser.add_argument("--tau", type=float, default=0.005,
                        help="Soft update coefficient")
    parser.add_argument("--buffer_capacity", type=int, default=1000000,
                        help="Replay buffer capacity")

    # Exploration
    parser.add_argument("--noise_scale", type=float, default=0.5,
                        help="Initial exploration noise scale")
    parser.add_argument("--noise_decay", type=float, default=0.999,
                        help="Noise decay rate per step")
    parser.add_argument("--warmup_steps", type=int, default=2000,
                        help="Random exploration steps before learning starts")

    # Logging & Saving
    parser.add_argument("--log_dir", type=str, default="./logs/maddpg",
                        help="Log directory")
    parser.add_argument("--save_dir", type=str, default="./checkpoints/maddpg",
                        help="Checkpoint save directory")
    parser.add_argument("--save_interval", type=int, default=500,
                        help="Save checkpoint every N episodes")
    parser.add_argument("--eval_interval", type=int, default=100,
                        help="Evaluate every N episodes")
    parser.add_argument("--eval_episodes", type=int, default=32,
                        help="Number of evaluation episodes")

    # Other
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
