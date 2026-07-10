# pogema_env.py
import numpy as np
from pogema import pogema_v0, GridConfig
from config import NUM_AGENTS, GRID_SIZE, OBSTACLE_DENSITY, OBS_RADIUS, MAX_EPISODE_STEPS, STEP_PENALTY, COLLISION_PENALTY


class PogemaWrapper:
    # обертка над POGEMA (п.2.2.1 задания): в самой среде награда только +1 за
    # цель и 0 в остальное время, без штрафа за шаг и за столкновения - оба
    # добавляем здесь. Также даем удобные геттеры для визуализации (координаты,
    # препятствия, глобальное состояние для гиперсетей QMIX / критика MADDPG)
    def __init__(self, seed=None):
        self.grid_config = GridConfig(
            num_agents=NUM_AGENTS,
            size=GRID_SIZE,
            density=OBSTACLE_DENSITY,
            obs_radius=OBS_RADIUS,
            max_episode_steps=MAX_EPISODE_STEPS,
            seed=seed,
        )
        self.env = pogema_v0(grid_config=self.grid_config)
        self.n_agents = NUM_AGENTS
        self.obs_shape = self.env.observation_space.shape  # (3, 2*OBS_RADIUS+1, 2*OBS_RADIUS+1)
        self.n_actions = self.env.action_space.n  # 5

        self._active = [True] * self.n_agents
        self._prev_xy = None

    def reset(self, seed=None):
        obs, info = self.env.reset(seed=seed)
        self._active = [True] * self.n_agents
        self._prev_xy = self.env.get_agents_xy()
        return obs

    def step(self, actions):
        was_active = list(self._active)
        obs, rewards, terminated, truncated, info = self.env.step(actions)
        xy = self.env.get_agents_xy()

        shaped_rewards = []
        collisions = 0
        for i in range(self.n_agents):
            if not was_active[i]:
                # агент уже дошел до цели раньше - его позиция и так заморожена,
                # это не шаг и не столкновение, просто больше не учим его
                shaped_rewards.append(0.0)
                continue

            r = rewards[i] + STEP_PENALTY
            # столкновение: агент выбрал не "стоять", но фактически не сдвинулся -
            # значит уперся в препятствие или другого агента
            if actions[i] != 0 and xy[i] == self._prev_xy[i] and not terminated[i]:
                r += COLLISION_PENALTY
                collisions += 1
            shaped_rewards.append(r)
            self._active[i] = not terminated[i]

        self._prev_xy = xy
        done = [t or tr for t, tr in zip(terminated, truncated)]
        info_out = {"collisions": collisions, "raw_info": info}
        return obs, shaped_rewards, terminated, truncated, done, info_out

    def get_global_state(self):
        # плоский вектор глобального состояния для гиперсетей QMIX / центрального критика
        return np.asarray(self.env.grid.get_state(), dtype=np.float32)

    def get_agents_xy(self):
        return self.env.get_agents_xy()

    def get_targets_xy(self):
        return self.env.get_targets_xy()

    def get_obstacles(self):
        return np.array(self.env.grid.get_obstacles())

    def close(self):
        pass


def make_env(seed=None):
    return PogemaWrapper(seed=seed)
