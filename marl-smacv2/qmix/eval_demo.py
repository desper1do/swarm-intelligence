"""
Демонстрация работы обученной QMIX модели.

Пример:
    python eval_demo.py --checkpoint ./checkpoints/qmix/qmix_10gen_terran_ep500.pt --n_episodes 5
"""

import os
import sys
import argparse
import time
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from common.starcraft_env import SMACv2Wrapper
from common.runner import set_seed, get_device
from qmix.qmix_agent import QMIXAgent


def run_demo(args):
    set_seed(args.seed)
    device = get_device()

    env = SMACv2Wrapper(
        map_name=args.map_name,
        debug=False,
        conic_fov=args.conic_fov,
        use_unit_ranges=True,
        headless=False,
        step_mul=args.step_mul,
        seed=args.seed,
    )

    env_info = env.get_env_info()
    print(f"Environment: {env_info}")

    agent = QMIXAgent(
        n_agents=env_info["n_agents"],
        obs_dim=env_info["obs_dim"],
        n_actions=env_info["n_actions"],
        state_dim=env_info["state_dim"],
        device=device,
    )

    print(f"Loading checkpoint: {args.checkpoint}")
    agent.load(args.checkpoint)

    total_reward = 0
    wins = 0
    total_steps = 0

    for ep in range(1, args.n_episodes + 1):
        observations, state = env.reset()
        avail_actions = env.get_avail_actions()
        hidden_states = None
        episode_reward = 0
        step = 0

        print(f"\n--- Episode {ep}/{args.n_episodes} ---")

        for step in range(env.episode_limit):
            obs_t = torch.FloatTensor(observations).to(device)
            avail_t = torch.FloatTensor(avail_actions).to(device)
            actions, hidden_states = agent.select_actions(obs_t, avail_t, hidden_states, epsilon=0.0)

            observations, state, reward, terminated, info = env.step(actions)
            avail_actions = env.get_avail_actions()
            episode_reward += reward

            time.sleep(args.step_delay)

            if terminated:
                won = info.get("battle_won", False)
                if won:
                    wins += 1
                dead_a = info.get("dead_allies", 0)
                dead_e = info.get("dead_enemies", 0)
                print(f"  Reward: {episode_reward:.2f} | Steps: {step+1} | "
                      f"Win: {won} | Dead allies: {dead_a} | Dead enemies: {dead_e}")
                break

        total_reward += episode_reward
        total_steps += step + 1

    print(f"\n{'='*50}")
    print(f"Demo completed: {args.n_episodes} episodes")
    print(f"Avg reward: {total_reward/args.n_episodes:.2f}")
    print(f"Win rate: {wins/args.n_episodes:.1%}")
    print(f"Avg steps: {total_steps/args.n_episodes:.1f}")
    print(f"{'='*50}")

    env.close()


def main():
    parser = argparse.ArgumentParser(description="Run QMIX demo with trained model")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint file")
    parser.add_argument("--map_name", type=str, default="10gen_terran", help="Map name")
    parser.add_argument("--conic_fov", action="store_true", help="Use conic field of view")
    parser.add_argument("--n_episodes", type=int, default=5, help="Number of demo episodes")
    parser.add_argument("--step_delay", type=float, default=0.05, help="Delay between steps (seconds)")
    parser.add_argument("--step_mul", type=int, default=8, help="Game steps per agent step")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    run_demo(args)


if __name__ == "__main__":
    main()
