"""
Скрипт обучения MAPPO в SMACv2.

Пример запуска:
    cd mappo
    python train_mappo.py --map_name 10gen_terran --n_episodes 5000 --rollout_steps 400
"""

import os
import sys
import argparse
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from common.starcraft_env import SMACv2Wrapper
from common.runner import Logger, set_seed, get_device
from common.replay_buffer import RolloutBuffer
from mappo.mappo_agent import MAPPOAgent


def train(args):
    """Основной цикл обучения MAPPO."""
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

    # Создаём агента MAPPO
    agent = MAPPOAgent(
        n_agents=env_info["n_agents"],
        obs_dim=env_info["obs_dim"],
        n_actions=env_info["n_actions"],
        state_dim=env_info["state_dim"],
        hidden_dim=args.hidden_dim,
        rnn_hidden_dim=args.rnn_hidden_dim,
        lr_actor=args.lr_actor,
        lr_critic=args.lr_critic,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_range=args.clip_range,
        value_clip=args.value_clip,
        entropy_coef=args.entropy_coef,
        value_loss_coef=args.value_loss_coef,
        device=device,
    )

    # Rollout Buffer
    rollout_buffer = RolloutBuffer(
        n_steps=args.rollout_steps,
        n_agents=env_info["n_agents"],
        obs_dim=env_info["obs_dim"],
        state_dim=env_info["state_dim"],
        action_dim=env_info["n_actions"],
        device=device,
    )

    # Логгер
    logger = Logger(args.log_dir, f"mappo_{args.map_name}")
    logger.log(f"Starting MAPPO training on {args.map_name}")
    logger.log(f"Device: {device}")
    logger.log(f"Args: {args}")

    # Скрытые состояния RNN
    actor_hidden = None
    critic_hidden = None

    episode = 0

    try:
        while episode < args.n_episodes:
            # === Сбор данных ===
            rollout_buffer.clear()

            observations, state = env.reset()
            avail_actions = env.get_avail_actions()
            actor_hidden = None
            critic_hidden = None

            episode_reward = 0
            episode_steps = 0
            won = False

            for step in range(args.rollout_steps):
                obs_tensor = torch.FloatTensor(observations).to(device)
                state_tensor = torch.FloatTensor(state).to(device)
                avail_tensor = torch.FloatTensor(avail_actions).to(device)

                # Выбираем действия
                actions, log_probs, actor_hidden = agent.select_actions(
                    obs_tensor,
                    avail_tensor,
                    hidden_states=actor_hidden,
                    deterministic=False,
                )

                # Получаем value
                value, critic_hidden = agent.get_values(state_tensor, critic_hidden)

                # Выполняем шаг
                next_observations, next_state, reward, terminated, info = env.step(actions)
                next_avail_actions = env.get_avail_actions()

                # Сохраняем в буфер
                rollout_buffer.add(
                    obs=observations,
                    state=state,
                    action=actions,
                    log_prob=log_probs.cpu().numpy(),
                    reward=reward,
                    value=value.detach().cpu().numpy(),
                    done=terminated,
                    avail_actions=avail_actions,
                )

                episode_reward += reward
                observations = next_observations
                state = next_state
                avail_actions = next_avail_actions
                episode_steps += 1

                if terminated:
                    won = info.get("battle_won", False)
                    break

            episode += 1

            # === Обучение ===
            losses = {}
            if rollout_buffer.size > 0:
                losses = agent.train(
                    rollout_buffer,
                    n_epochs=args.n_epochs,
                    batch_size=args.batch_size,
                )

            # === Логирование ===
            logger.log_episode(
                episode,
                episode_reward,
                episode_steps,
                won=won,
                extra_info=losses,
            )

            # Сохраняем чекпоинт
            if episode % args.save_interval == 0:
                os.makedirs(args.save_dir, exist_ok=True)
                save_path = os.path.join(args.save_dir, f"mappo_{args.map_name}_ep{episode}.pt")
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
        save_path = os.path.join(args.save_dir, f"mappo_{args.map_name}_interrupted.pt")
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
        actor_hidden = None
        episode_reward = 0

        for step in range(env.episode_limit):
            obs_tensor = torch.FloatTensor(observations).to(device)
            avail_tensor = torch.FloatTensor(avail_actions).to(device)

            actions, _, actor_hidden = agent.select_actions(
                obs_tensor,
                avail_tensor,
                hidden_states=actor_hidden,
                deterministic=True,
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
    parser = argparse.ArgumentParser(description="Train MAPPO on SMACv2")

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
    parser.add_argument("--rollout_steps", type=int, default=400,
                        help="Steps per rollout")
    parser.add_argument("--n_epochs", type=int, default=10,
                        help="PPO epochs per update")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--gae_lambda", type=float, default=0.95, help="GAE lambda")

    # Model
    parser.add_argument("--hidden_dim", type=int, default=64, help="Hidden dimension")
    parser.add_argument("--rnn_hidden_dim", type=int, default=64,
                        help="RNN hidden dimension")
    parser.add_argument("--lr_actor", type=float, default=5e-4,
                        help="Actor learning rate")
    parser.add_argument("--lr_critic", type=float, default=5e-4,
                        help="Critic learning rate")
    parser.add_argument("--clip_range", type=float, default=0.2,
                        help="PPO clip range")
    parser.add_argument("--value_clip", action="store_true", default=True,
                        help="Clip value loss")
    parser.add_argument("--entropy_coef", type=float, default=0.01,
                        help="Entropy coefficient")
    parser.add_argument("--value_loss_coef", type=float, default=0.5,
                        help="Value loss coefficient")

    # Logging & Saving
    parser.add_argument("--log_dir", type=str, default="./logs/mappo",
                        help="Log directory")
    parser.add_argument("--save_dir", type=str, default="./checkpoints/mappo",
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
