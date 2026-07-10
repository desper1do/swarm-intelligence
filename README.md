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

на среде FrozenLake-v1 (Gymnasium).

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
