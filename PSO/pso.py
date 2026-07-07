import numpy as np
from config import V_MAX, W, C1, C2, SEARCH_MIN, SEARCH_MAX, VELOCITY_MIN, VELOCITY_MAX, N_PARTICLES


def initialize_swarm(objective_function):
    # Шаг 0: инициализация роя
    # Случайные позиции и скорости частиц в пределах области поиска
    position = np.random.uniform(SEARCH_MIN, SEARCH_MAX, size=(N_PARTICLES, 2))
    velocity = np.random.uniform(VELOCITY_MIN, VELOCITY_MAX, size=(N_PARTICLES, 2))

    # Шаг 1: оценка для только что созданного роя личный лучший результат
    # pbest каждой частицы - ее стартовая позиция
    pbest_position = position.copy()
    x = position[:, 0]
    y = position[:, 1]
    pbest_value = objective_function(x, y)

    # начальный gbest (лучшая из pbest по всему рою)
    best_index = np.argmin(pbest_value)
    gbest_value = pbest_value[best_index]
    gbest_position = pbest_position[best_index].copy()

    return position, velocity, pbest_position, pbest_value, gbest_position, gbest_value


def update_swarm(position, velocity, pbest_position, pbest_value, gbest_position, gbest_value, objective_function):
    # Одна итерация PSO: пересчёт скорости и позиции всех частиц,
    # затем обновление pbest/gbest по уравнениям Кеннеди-Эберхарта

    # r1, r2 - случайные числа из [0, 1], вносят стохастичность в поиск
    r1 = np.random.rand(N_PARTICLES, 2)
    r2 = np.random.rand(N_PARTICLES, 2)

    # Шаг 4: обновление скорости
    # инерция + когнитивная составляющая (притяжение к pbest) + социальная составляющая (притяжение к gbest)
    velocity = W*velocity + C1*r1*(pbest_position - position) + C2*r2*(gbest_position - position)
    velocity = np.clip(velocity, -V_MAX, V_MAX)  # ограничение скорости, чтобы рой не разлетался

    # Шаг 5: обновление позиции частиц по новой скорости
    position = position + velocity
    # Шаг 6: проверка ограничений (не выходить за пределы области поиска)
    position = np.clip(position, SEARCH_MIN, SEARCH_MAX)

    # Шаг 1 (повторно): оценка целевой функции в новых позициях
    current_value = objective_function(position[:, 0], position[:, 1])

    for i in range(N_PARTICLES):
        # Шаг 2: обновление pbest (если частица улучшила свой личный результат)
        if current_value[i] < pbest_value[i]:
            pbest_position[i] = position[i].copy()
            pbest_value[i] = current_value[i]
            # Шаг 3: обновление gbest (если это ещё и лучший результат всего роя)
            if pbest_value[i] < gbest_value:
                gbest_value = pbest_value[i]
                gbest_position = pbest_position[i].copy()

    return position, velocity, pbest_position, pbest_value, gbest_position, gbest_value