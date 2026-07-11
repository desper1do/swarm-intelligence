import numpy as np
import torch
import gymnasium as gym

from config import SEED
from train import train_all
from compare import plot_comparison
from demo import play_agent, play_reinforce

if __name__ == "__main__":
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    train_env = gym.make("Taxi-v4")
    results = train_all(train_env)
    train_env.close()

    plot_comparison(results)

    demo_env = gym.make("Taxi-v4", render_mode="human")
    play_agent(demo_env, results["sarsa"]["agent"], episodes=3, delay=0.3)
    demo_env.close()

    demo_env = gym.make("Taxi-v4", render_mode="human")
    play_agent(demo_env, results["qlearning"]["agent"], episodes=3, delay=0.3)
    demo_env.close()

    demo_env = gym.make("Taxi-v4", render_mode="human")
    play_reinforce(demo_env, results["reinforce"]["agent"], episodes=3, delay=0.3)
    demo_env.close()