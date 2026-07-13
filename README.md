# Роевой интеллект

Репозиторий с практическими заданиями по дисциплине «Роевой интеллект»
(УрФУ, ИРИТ-РтФ, преподаватель Третьяков С.А.).

Автор: Насибулин Данила Евгеньевич, РИ-240943

Реализованы классические алгоритмы роевого интеллекта:

- **PSO** — Particle Swarm Optimization
- **ACO** — Ant Colony Optimization

Также реализованы классические алгоритмы обучения с подкреплением:

- **Q-Learning** (off-policy)
- **SARSA** (on-policy)
- **REINFORCE** (policy-gradient)

на средах FrozenLake-v1 (`frozenlake-rl/`), CliffWalking-v1 (`cliffwalking-rl/`)
и Taxi-v4 (`taxi-v3/`, Лабораторная работа №1) из Gymnasium.

И алгоритмы глубокого мультиагентного обучения с подкреплением:

- **QMIX** — value-based, монотонная факторизация Q через смешивающую сеть
- **MADDPG** — policy-gradient, actor-critic с централизованными критиками
- **MAPPO** — on-policy actor-critic (PPO) с parameter sharing

на средах POGEMA (`pogema-marl/`, кооперативный мультиагентный поиск пути)
и SMACv2 / StarCraft II (`marl-smacv2/`, Лабораторная работа №2).

## Запуск

Требуется Python 3. Установка окружения:

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

В каждой папке (`PSO/`, `ACO/`) есть `main.py` — запускает анимацию
работы алгоритма и выводит в консоль прогресс оптимизации по итерациям:

```bash
python PSO/main.py
python ACO/main.py
```

Во время анимации пробел ставит её на паузу/возобновляет.

`frozenlake-rl/main.py` обучает Q-Learning и SARSA на FrozenLake-v1,
показывает анимацию обучения, сохраняет графики в `frozenlake-rl/plots/`
и демонстрирует выученную политику в отдельном окне:

```bash
python frozenlake-rl/main.py
```

`cliffwalking-rl/main.py` — то же самое (SARSA как основной алгоритм,
Q-Learning для сравнения) на среде CliffWalking-v1, графики в
`cliffwalking-rl/plots/`:

```bash
python cliffwalking-rl/main.py
```

## Лабораторная работа №1: Taxi-v4 (Q-Learning / SARSA / REINFORCE)

`taxi-v3/main.py` обучает все три алгоритма (Q-Learning, SARSA, REINFORCE)
на Taxi-v4, строит сравнительные графики (`taxi-v3/plots/`) и показывает
демонстрацию выученных политик в отдельном окне:

```bash
python taxi-v3/main.py
```

Дополнительные скрипты для самостоятельной части: `experiments.py`
(влияние гиперпараметров: alpha, epsilon decay), `task1_comparison.py`
(сравнение алгоритмов), `analyze_policy.py` (разбор выученной policy).
Скриншоты кода/логов/демо — в `taxi-v3/screenshots/`.

## Мультиагентное обучение (POGEMA)

`pogema-marl/` использует **отдельное виртуальное окружение внутри своей
папки**, а не корневое: пакет `pogema` требует `gymnasium==0.28.1` и
`numpy<=1.26.4`, что конфликтует с остальными практическими. Поэтому зависимости
и venv у него свои (см. `pogema-marl/requirements.txt`):

```bash
cd pogema-marl
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python main.py
```

`main.py` обучает QMIX и MADDPG на POGEMA с анимацией в реальном времени
(карта + график сходимости), затем показывает зацикленный повтор эволюции
траекторий по эпизодам и сохраняет все графики (сходимость, success rate,
epsilon/шум, тепловые карты, сравнение алгоритмов) в `pogema-marl/plots/`.
Обучение идёт на GPU (CUDA), если доступен, иначе на CPU.

## Лабораторная работа №2: MARL на SMACv2 (StarCraft II)

`marl-smacv2/` реализует QMIX, MADDPG и MAPPO (парадигма CTDE — Centralized
Training with Decentralized Execution) для среды **SMACv2** (StarCraft
Multi-Agent Challenge v2, StarCraft II). Как и `pogema-marl/`, использует
**отдельное виртуальное окружение внутри своей папки** и требует отдельно
установленный StarCraft II + карты SMAC (см. подробную инструкцию по
установке в `marl-smacv2/README.md`).

```bash
cd marl-smacv2
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install torch numpy
pip install -e /path/to/smacv2  # git clone https://github.com/oxwhirl/smacv2

python qmix/train_qmix.py --map_name 3m --n_episodes 150
python maddpg/train_maddpg.py --map_name 3m --n_episodes 150
python mappo/train_mappo.py --map_name 3m --n_episodes 150
```

Каждый `train_*.py` пишет метрики эпизодов в `logs/<algo>/` (JSON) и
чекпоинты в `checkpoints/<algo>/`. `plot_results.py` строит сравнительные
графики обучения по нескольким JSON-логам сразу:

```bash
python plot_results.py --log logs/qmix/qmix_3m_metrics.json logs/maddpg/maddpg_3m_metrics.json logs/mappo/mappo_3m_metrics.json \
    --labels QMIX MADDPG MAPPO --save_dir plots/compare_3m
```

`eval_demo.py` в каждой папке алгоритма запускает визуальную демонстрацию
обученного чекпоинта в реальном окне StarCraft II (не headless):

```bash
python qmix/eval_demo.py --checkpoint checkpoints/qmix/qmix_3m_ep150.pt --map_name 3m --n_episodes 3
```

Результаты обучения (логи, чекпоинты, графики) не версионируются
(см. `.gitignore`) — воспроизводятся запуском скриптов выше.