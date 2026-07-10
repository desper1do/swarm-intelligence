import gymnasium as gym
from config import MAP_NAME, IS_SLIPPERY

def make_env(render_mode=None):
    # is_slippery=True - агент может соскользнуть в сторону от намеченного
    # направления (вероятность 1/3 на каждое из двух перпендикулярных направлений)
    env = gym.make(
        "FrozenLake-v1",
        map_name=MAP_NAME,
        is_slippery=IS_SLIPPERY,
        render_mode=render_mode,
    )
    return env

def get_env_info(env):
    # |S| и |A| - размерности Q-таблицы
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    return n_states, n_actions

