"""
SMACv2 Environment Wrapper для MARL алгоритмов.
Поддерживает как стандартный SMACv2 API, так и удобную обёртку для PyTorch.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional

from smacv2.env import StarCraft2Env
from smacv2.env.starcraft2.wrapper import StarCraftCapabilityEnvWrapper


def _fast_restart(self):
    """Патч StarCraft2Env._restart(): используем full_restart() (перезапуск процесса SC2,
    ~7-10с) вместо штатного _kill_all_units() (циклический debug-kill + опрос, пока юниты
    не исчезнут). Диагностика (diag_full_restart.py) показала, что после эпизода с реальным
    боем этот цикл может занимать 500+с (сотни тысяч итераций step()/observe(), каждая быстрая
    сама по себе, но их совокупно очень много), тогда как full_restart() стабильно укладывается
    в единицы секунд. Наблюдалось независимо от networkingMode WSL2 (mirrored/NAT)."""
    self.full_restart()


StarCraft2Env._restart = _fast_restart


def _is_generative_map(map_name: str) -> bool:
    """10gen_* карты используют процедурную генерацию состава (capability_config).
    Классические карты (3m, 8m, 2s3z, ...) имеют фиксированный состав и не
    поддерживают capability_config вообще."""
    return "gen_" in map_name


class SMACv2Wrapper:
    """
    Удобная обёртка для SMACv2 среды.
    Преобразует данные в PyTorch-совместимый формат.
    """

    def __init__(
        self,
        map_name: str = "10gen_terran",
        capability_config: Optional[Dict] = None,
        debug: bool = False,
        conic_fov: bool = False,
        use_unit_ranges: bool = True,
        min_attack_range: int = 2,
        obs_own_pos: bool = True,
        fully_observable: bool = False,
        headless: bool = True,
        step_mul: int = 8,
        seed: int = 42,
    ):
        """
        Args:
            step_mul: Количество игровых шагов SC2 за один агентский шаг.
                      Больше значение = быстрее обучение, но грубее симуляция.
                      Рекомендуемые: 8 (default), 16, 32 (максимальная скорость).
            headless: Если True, запускает SC2 в минимальном окне 1x1 пиксель
                      (отключает GPU-рендеринг, но процесс SC2 всё равно запускается).
        """
        self.map_name = map_name
        self.seed = seed

        # Headless mode: минимальный window для отключения GPU-рендеринга
        window_kwargs = {}
        if headless:
            window_kwargs["window_size_x"] = 1
            window_kwargs["window_size_y"] = 1

        if _is_generative_map(map_name):
            if capability_config is None:
                capability_config = self._default_capability_config(map_name)
            self.env = StarCraftCapabilityEnvWrapper(
                capability_config=capability_config,
                map_name=map_name,
                debug=debug,
                conic_fov=conic_fov,
                use_unit_ranges=use_unit_ranges,
                min_attack_range=min_attack_range,
                obs_own_pos=obs_own_pos,
                fully_observable=fully_observable,
                step_mul=step_mul,
                **window_kwargs,
            )
        else:
            # Классическая SMAC-карта с фиксированным составом юнитов
            # (3m, 8m, 2s3z, ...) — capability_config здесь не поддерживается.
            self.env = StarCraft2Env(
                map_name=map_name,
                debug=debug,
                obs_own_pos=obs_own_pos,
                step_mul=step_mul,
                **window_kwargs,
            )

        env_info = self.env.get_env_info()
        self.n_agents = env_info["n_agents"]
        self.n_actions = env_info["n_actions"]
        self.state_dim = env_info["state_shape"]
        self.obs_dim = env_info["obs_shape"]
        self.episode_limit = env_info["episode_limit"]
        self.cap_shape = env_info.get("cap_shape", 0)

        self._episode_count = 0
        self._episode_steps = 0
        self._terminated = False

    def _default_capability_config(self, map_name: str) -> Dict:
        """Создаёт конфигурацию по умолчанию на основе имени карты."""
        # Определяем количество юнитов из имени карты
        parts = map_name.split("_")
        n_units = 10
        n_enemies = 11
        try:
            if parts[0].endswith("gen"):
                n_units = int(parts[0].replace("gen", ""))
            else:
                n_units = int(parts[0])
        except (ValueError, IndexError):
            pass

        return {
            "n_units": n_units,
            "n_enemies": n_enemies,
            "team_gen": {
                "dist_type": "weighted_teams",
                "unit_types": ["marine", "marauder", "medivac"],
                "weights": [0.45, 0.45, 0.1],
                "observe": True,
                "exception_unit_types": ["medivac"],
            },
            "start_positions": {
                "dist_type": "surrounded_and_reflect",
                "p": 0.5,
                "map_x": 32,
                "map_y": 32,
            },
        }

    def reset(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Сбрасывает среду.

        Returns:
            observations: (n_agents, obs_dim)
            state: (state_dim,)
        """
        self._terminated = False
        self._episode_steps = 0
        self._episode_count += 1

        obs_list, state_list = self.env.reset()
        observations = np.array(obs_list, dtype=np.float32)
        state = np.array(state_list, dtype=np.float32)

        return observations, state

    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float, bool, Dict]:
        """
        Выполняет шаг в среде.

        Args:
            actions: (n_agents,) - действия для каждого агента

        Returns:
            observations: (n_agents, obs_dim)
            state: (state_dim,)
            reward: скаляр
            terminated: флаг завершения эпизода
            info: дополнительная информация
        """
        actions = actions.astype(np.int64).tolist()
        reward, terminated, info = self.env.step(actions)

        self._terminated = terminated
        self._episode_steps += 1

        # Если эпизод не завершён, получаем новые наблюдения
        if not terminated:
            obs_list = self.env.get_obs()
            state_list = self.env.get_state()
            observations = np.array(obs_list, dtype=np.float32)
            state = np.array(state_list, dtype=np.float32)
        else:
            observations = np.zeros((self.n_agents, self.obs_dim), dtype=np.float32)
            state = np.zeros(self.state_dim, dtype=np.float32)

        return observations, state, float(reward), terminated, info

    def get_avail_actions(self) -> np.ndarray:
        """
        Возвращает доступные действия для всех агентов.

        Returns:
            avail_actions: (n_agents, n_actions) - маска доступных действий
        """
        avail_actions = []
        for agent_id in range(self.n_agents):
            avail = self.env.get_avail_agent_actions(agent_id)
            avail_actions.append(avail)
        return np.array(avail_actions, dtype=np.float32)

    def get_env_info(self) -> Dict:
        """Возвращает информацию о среде."""
        return {
            "n_agents": self.n_agents,
            "n_actions": self.n_actions,
            "state_dim": self.state_dim,
            "obs_dim": self.obs_dim,
            "episode_limit": self.episode_limit,
            "cap_shape": self.cap_shape,
        }

    def close(self):
        """Закрывает среду."""
        self.env.close()

    @property
    def terminated(self) -> bool:
        return self._terminated

    @property
    def episode_steps(self) -> int:
        return self._episode_steps
