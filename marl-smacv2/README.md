# MARL Algorithms for SMACv2

Реализация алгоритмов мультиагентного обучения с подкреплением (MARL) для среды **SMACv2** (StarCraft Multi-Agent Challenge v2):

- **MADDPG** — Multi-Agent Deep Deterministic Policy Gradient (off-policy, actor-critic)
- **QMIX** — Q-value MIXing с DRQN (off-policy, value-based)
- **MAPPO** — Multi-Agent Proximal Policy Optimization (on-policy, actor-critic)

Все алгоритмы реализуют парадигму **CTDE** (Centralized Training with Decentralized Execution).

---

## Установка

### 1. Установка StarCraft II

Скачайте StarCraft II с [официального сайта](https://starcraft2.com/) и установите в `~/StarCraftII/`:

```bash
# Или используйте скрипт (после установки smacv2)
bash smacv2/bin/install_sc2.sh
```

### 2. Установка Python-зависимостей

```bash
# SMACv2
git clone https://github.com/oxwhirl/smacv2.git
cd smacv2
pip install -e .

# PyTorch и numpy
pip install torch numpy
```

### 3. Структура проекта

```
marl_smacv2/
  common/               # Общие модули
    starcraft_env.py    # Wrapper для SMACv2
    networks.py         # Нейросетевые архитектуры
    replay_buffer.py    # Буферы данных
    runner.py           # Утилиты для запуска и логирования
  maddpg/               # MADDPG
    maddpg_agent.py     # Реализация
    train_maddpg.py     # Скрипт обучения
  qmix/                 # QMIX
    qmix_agent.py       # Реализация
    train_qmix.py       # Скрипт обучения
  mappo/                # MAPPO
    mappo_agent.py      # Реализация
    train_mappo.py      # Скрипт обучения
  README.md
```

---

## Быстрый старт

### MADDPG

```bash
cd maddpg
python train_maddpg.py --map_name 10gen_terran --n_episodes 5000 --batch_size 256 --lr_actor 1e-4 --lr_critic 1e-3
```

### QMIX

```bash
cd qmix
python train_qmix.py --map_name 10gen_terran --n_episodes 5000 --batch_size 32 --lr 5e-4
```

### MAPPO

```bash
cd mappo
python train_mappo.py --map_name 10gen_terran --n_episodes 5000 --rollout_steps 400 --lr_actor 5e-4 --lr_critic 5e-4
```

---

## Доступные карты SMACv2

| Карта | Описание |
|-------|----------|
| `3m` | 3 марина vs 3 марина |
| `8m` | 8 маринов vs 8 маринов |
| `25m` | 25 маринов vs 25 маринов |
| `2s3z` | 2 сталкера + 3 зерлинга |
| `3s5z` | 3 сталкера + 5 зерлингов |
| `1c3s5z` | 1 колосс + 3 сталкера + 5 зерлингов |
| `10gen_terran` | 10 терран юнитов (генерируемая) |
| `10gen_zerg` | 10 зерг юнитов (генерируемая) |
| `10gen_protoss` | 10 протосс юнитов (генерируемая) |
| `MMM` | Смешанный состав |

---

## Параметры алгоритмов

### MADDPG

| Параметр | Описание | По умолчанию |
|----------|----------|-------------|
| `--lr_actor` | Learning rate актора | 1e-4 |
| `--lr_critic` | Learning rate критика | 1e-3 |
| `--tau` | Коэффициент soft update | 0.005 |
| `--noise_scale` | Начальный шум исследования | 0.3 |
| `--noise_decay` | Затухание шума | 0.9995 |
| `--buffer_capacity` | Размер replay buffer | 1000000 |
| `--batch_size` | Размер батча | 256 |
| `--gamma` | Дисконт-фактор | 0.99 |

### QMIX

| Параметр | Описание | По умолчанию |
|----------|----------|-------------|
| `--lr` | Learning rate | 5e-4 |
| `--tau` | Soft update коэффициент | 0.005 |
| `--hidden_dim` | Размер скрытого слоя | 64 |
| `--rnn_hidden_dim` | RNN скрытое состояние | 64 |
| `--epsilon_start` | Начальный epsilon | 1.0 |
| `--epsilon_end` | Минимальный epsilon | 0.05 |
| `--epsilon_decay` | Затухание epsilon | 0.995 |
| `--buffer_capacity` | Размер буфера (в эпизодах) | 5000 |

### MAPPO

| Параметр | Описание | По умолчанию |
|----------|----------|-------------|
| `--lr_actor` | Learning rate актора | 5e-4 |
| `--lr_critic` | Learning rate критика | 5e-4 |
| `--gamma` | Дисконт-фактор | 0.99 |
| `--gae_lambda` | GAE lambda | 0.95 |
| `--clip_range` | PPO clip range | 0.2 |
| `--entropy_coef` | Коэффициент энтропии | 0.01 |
| `--rollout_steps` | Шагов в rollout | 400 |
| `--n_epochs` | PPO эпох за обновление | 10 |

---

## Архитектура алгоритмов

### MADDPG

- **Актор**: MLP `obs_dim -> [256, 128] -> n_actions` (tanh)
- **Критик**: MLP `n_agents * (obs + act) -> [256, 128] -> 1`
- **Исследование**: Ornstein-Uhlenbeck шум с затуханием
- **Обновление**: Soft update target сетей (tau=0.005)

### QMIX

- **Агент**: DRQN `obs -> GRU(64) -> [64] -> n_actions`
- **Mixer**: Hypernetwork `state -> w1, w2, b1, b2 -> Q_tot`
- **Исследование**: Epsilon-greedy (1.0 -> 0.05)
- **Обновление**: Double DQN + Soft update

### MAPPO

- **Актор**: RNN `obs -> GRU(64) -> [64] -> n_actions` (Categorical)
- **Критик**: RNN `state -> GRU(64) -> [64] -> 1` (централизованный)
- **Обучение**: PPO clipped objective + GAE
- **Особенности**: Parameter sharing, value clipping

---

## Логирование и чекпоинты

Все метрики сохраняются автоматически:

- **Логи**: `./logs/<algorithm>/<experiment>_log.txt`
- **Метрики JSON**: `./logs/<algorithm>/<experiment>_metrics.json`
- **Чекпоинты**: `./checkpoints/<algorithm>/`

Отслеживаются:
- Награда за эпизод и средняя награда (100 эпизодов)
- Win rate
- Losses обучения
- Epsilon / noise scale

При прерывании обучения (`Ctrl+C`) автоматически сохраняется чекпоинт.

---

## Пример использования в коде

```python
from common.starcraft_env import SMACv2Wrapper
from maddpg.maddpg_agent import MADDPGTrainer

# Создаём среду
env = SMACv2Wrapper(map_name="10gen_terran")
env_info = env.get_env_info()

# Создаём тренер
trainer = MADDPGTrainer(
    n_agents=env_info["n_agents"],
    obs_dim=env_info["obs_dim"],
    n_actions=env_info["n_actions"],
    state_dim=env_info["state_dim"],
    device="cuda",
)

# Обучение
for episode in range(1000):
    obs, state = env.reset()
    done = False

    while not done:
        actions = trainer.select_actions(obs, env.get_avail_actions(), explore=True)
        next_obs, next_state, reward, done, info = env.step(actions)
        trainer.store_transition(state, obs, actions, reward, next_state, next_obs, done)
        trainer.train()
        obs, state = next_obs, next_state

env.close()
```

---

## Лицензия

MIT
