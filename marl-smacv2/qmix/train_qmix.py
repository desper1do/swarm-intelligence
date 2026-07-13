"""
Скрипт обучения QMIX в SMACv2.

Пример запуска:
    cd qmix
    python train_qmix.py --map_name 10gen_terran --n_episodes 5000 --batch_size 32
"""

import os
import sys
import argparse
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from common.starcraft_env import SMACv2Wrapper
from common.runner import Logger, set_seed, get_device
from common.replay_buffer import EpisodeBuffer
from qmix.qmix_agent import QMIXAgent


def train(args):
    """Основной цикл обучения QMIX."""
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

    # Создаём агента QMIX
    agent = QMIXAgent(
        n_agents=env_info["n_agents"],
        obs_dim=env_info["obs_dim"],
        n_actions=env_info["n_actions"],
        state_dim=env_info["state_dim"],
        hidden_dim=args.hidden_dim,
        rnn_hidden_dim=args.rnn_hidden_dim,
        mixer_hidden_dim=args.mixer_hidden_dim,
        hypernet_hidden_dim=args.hypernet_hidden_dim,
        gamma=args.gamma,
        lr=args.lr,
        tau=args.tau,
        device=device,
    )

    # Replay Buffer
    replay_buffer = EpisodeBuffer(
        capacity=args.buffer_capacity,
        n_agents=env_info["n_agents"],
        obs_dim=env_info["obs_dim"],
        state_dim=env_info["state_dim"],
        n_actions=env_info["n_actions"],
        episode_limit=env_info["episode_limit"],
        device=device,
    )

    # Логгер
    logger = Logger(args.log_dir, f"qmix_{args.map_name}")
    logger.log(f"Starting QMIX training on {args.map_name}")
    logger.log(f"Device: {device}")
    logger.log(f"Args: {args}")

    # Epsilon для epsilon-greedy
    epsilon = args.epsilon_start

    try:
        # Основной цикл обучения
        for episode in range(1, args.n_episodes + 1):
            observations, state = env.reset()
            avail_actions = env.get_avail_actions()

            # Скрытые состояния RNN
            hidden_states = None

            episode_reward = 0
            won = False

            for step in range(env.episode_limit):
                # Выбираем действия
                obs_tensor = torch.FloatTensor(observations).to(device)
                avail_tensor = torch.FloatTensor(avail_actions).to(device)

                actions, hidden_states = agent.select_actions(
                    obs_tensor,
                    avail_tensor,
                    hidden_states=hidden_states,
                    epsilon=epsilon,
                )

                # Выполняем шаг
                next_observations, next_state, reward, terminated, info = env.step(actions)
                next_avail_actions = env.get_avail_actions()

                # Сохраняем шаг
                replay_buffer.add_step(
                    observations, state, actions, reward,
                    next_observations, next_state, terminated,
                    avail_actions, next_avail_actions,
                    padded=False,
                )

                episode_reward += reward
                observations = next_observations
                state = next_state
                avail_actions = next_avail_actions

                if terminated:
                    won = info.get("battle_won", False)
                    break

            # Добавляем паддинг для фиксированной длины эпизода
            if env.episode_steps < env.episode_limit:
                for pad_step in range(env.episode_steps, env.episode_limit):
                    replay_buffer.add_step(
                        np.zeros((env_info["n_agents"], env_info["obs_dim"]), dtype=np.float32),
                        np.zeros(env_info["state_dim"], dtype=np.float32),
                        np.zeros(env_info["n_agents"], dtype=np.int64),
                        0.0,
                        np.zeros((env_info["n_agents"], env_info["obs_dim"]), dtype=np.float32),
                        np.zeros(env_info["state_dim"], dtype=np.float32),
                        True,
                        np.ones((env_info["n_agents"], env_info["n_actions"]), dtype=np.float32),
                        np.ones((env_info["n_agents"], env_info["n_actions"]), dtype=np.float32),
                        padded=True,
                    )

            replay_buffer.end_episode()

            # Обучаем
            losses = {}
            if len(replay_buffer) >= args.batch_size:
                for _ in range(args.train_epochs):
                    batch = replay_buffer.sample(args.batch_size)
                    if batch is not None:
                        batch_losses = agent.train(batch)
                        for k, v in batch_losses.items():
                            losses[k] = losses.get(k, 0) + v
                losses = {k: v / args.train_epochs for k, v in losses.items()}

            # Логируем
            logger.log_episode(
                episode,
                episode_reward,
                env.episode_steps,
                won=won,
                extra_info={**losses, "epsilon": epsilon},
            )

            # Decay epsilon
            epsilon = max(args.epsilon_end, epsilon * args.epsilon_decay)

            # Сохраняем чекпоинт
            if episode % args.save_interval == 0:
                os.makedirs(args.save_dir, exist_ok=True)
                save_path = os.path.join(args.save_dir, f"qmix_{args.map_name}_ep{episode}.pt")
                agent.save(save_path)

            # Оценка
            if episode % args.eval_interval == 0:
                eval_reward, eval_win = evaluate(agent, env, args.eval_episodes, device)
                logger.log(
                    f"Eval after {episode} episodes | "
                    f"Reward: {eval_reward:.2f} | WinRate: {eval_win:.2%}"
                )

    except KeyboardInterrupt:
        logger.log("Training interrupted by user.")
        os.makedirs(args.save_dir, exist_ok=True)
        save_path = os.path.join(args.save_dir, f"qmix_{args.map_name}_interrupted.pt")
        agent.save(save_path)
        logger.log(f"Checkpoint saved: {save_path}")

    finally:
        env.close()
        logger.log("Environment closed. Exiting.")


def evaluate(agent, env, n_episodes, device):
    """Оценивает политику без исследования."""
    total_reward = 0
    wins = 0

    for _ in range(n_episodes):
        observations, state = env.reset()
        avail_actions = env.get_avail_actions()
        hidden_states = None
        episode_reward = 0

        for step in range(env.episode_limit):
            obs_tensor = torch.FloatTensor(observations).to(device)
            avail_tensor = torch.FloatTensor(avail_actions).to(device)

            actions, hidden_states = agent.select_actions(
                obs_tensor,
                avail_tensor,
                hidden_states=hidden_states,
                epsilon=0.0,  # Без исследования
            )

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
    parser = argparse.ArgumentParser(description="Train QMIX on SMACv2")

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
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Number of episodes per batch")
    parser.add_argument("--train_epochs", type=int, default=4,
                        help="Training epochs per episode")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")

    # Model
    parser.add_argument("--hidden_dim", type=int, default=64, help="Hidden dimension")
    parser.add_argument("--rnn_hidden_dim", type=int, default=64,
                        help="RNN hidden dimension")
    parser.add_argument("--mixer_hidden_dim", type=int, default=32,
                        help="Mixer hidden dimension")
    parser.add_argument("--hypernet_hidden_dim", type=int, default=64,
                        help="Hypernet hidden dimension")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--tau", type=float, default=0.005,
                        help="Soft update coefficient")
    parser.add_argument("--buffer_capacity", type=int, default=5000,
                        help="Buffer capacity in episodes")

    # Exploration
    parser.add_argument("--epsilon_start", type=float, default=1.0,
                        help="Initial epsilon for exploration")
    parser.add_argument("--epsilon_end", type=float, default=0.05,
                        help="Final epsilon")
    parser.add_argument("--epsilon_decay", type=float, default=0.995,
                        help="Epsilon decay rate per episode")

    # Logging & Saving
    parser.add_argument("--log_dir", type=str, default="./logs/qmix",
                        help="Log directory")
    parser.add_argument("--save_dir", type=str, default="./checkpoints/qmix",
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
