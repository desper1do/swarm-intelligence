import gymnasium as gym
from config import ENV_ID, IS_SLIPPERY

def make_env(render_mode=None):
    # is_slippery=False переходы детерминированы, действие всегда приводит
    # к ожидаемой соседней клетке (в отличие от FrozenLake)
    env = gym.make(
        ENV_ID,
        is_slippery=IS_SLIPPERY,
        render_mode=render_mode,
    )
    return env

def get_env_info(env):
    # |S| и |A| размерности Q-таблицы (48 состояний на сетке 4x12, 4 действия)
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    return n_states, n_actions
