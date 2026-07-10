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

на средах FrozenLake-v1 (`frozenlake-rl/`) и CliffWalking-v1 (`cliffwalking-rl/`)
из Gymnasium.

И алгоритмы глубокого мультиагентного обучения с подкреплением (`pogema-marl/`):

- **QMIX** — value-based, монотонная факторизация Q через смешивающую сеть
- **MADDPG** — policy-gradient, actor-critic с централизованными критиками

на среде POGEMA (кооперативный мультиагентный поиск пути).

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
