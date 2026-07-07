import numpy as np
from config import N_CITIES, M_ANTS, ALPHA, BETA, RHO, Q, TAU0


def initialize_aco(distances):
    # Шаг 1: инициализация феромонов и эвристической информации
    # tau0 одинаковый для всех ребер, эвристика - обратная величина расстояния (ближе город, привлекательнее)
    pheromone = np.full((N_CITIES, N_CITIES), TAU0)
    heuristic = 1.0 / distances
    best_route = None
    best_length = np.inf
    return pheromone, heuristic, best_route, best_length


def construct_solutions(pheromone, heuristic, distances):
    # Шаг 2: конструирование решения, каждый муравей строит маршрут по вероятностному правилу
    routes = np.zeros((M_ANTS, N_CITIES), dtype=int)

    for ant in range(M_ANTS):
        start_city = np.random.randint(N_CITIES)  # муравей стартует со случайного города
        visited = [start_city]

        for _ in range(N_CITIES - 1):
            current = visited[-1]
            # вероятность перехода Pij ~ (феромон^alpha) * (эвристика^beta)
            probs = (pheromone[current] ** ALPHA) * (heuristic[current] ** BETA)
            probs[visited] = 0  # посещенные города исключаем из выбора
            probs = probs / probs.sum()  # нормируем в распределение вероятностей

            next_city = np.random.choice(N_CITIES, p=probs)
            visited.append(next_city)

        routes[ant] = visited

    return routes


def calculate_route_length(route, distances):
    # Шаг 3: длина одного маршрута (замкнутый цикл, последний город соединен с первым)
    length = 0.0
    for i in range(len(route)):
        current_city = route[i]
        next_city = route[(i + 1) % len(route)]
        length += distances[current_city, next_city]
    return length


def calculate_all_lengths(routes, distances):
    # длина маршрута каждого муравья
    return np.array([calculate_route_length(route, distances) for route in routes])


def update_pheromones(pheromone, routes, lengths):
    # Шаг 5: испарение феромона на всех ребрах
    pheromone *= (1 - RHO)

    # Шаг 4: глобальное обновление, феромон добавляется только на ребра лучшего маршрута итерации
    best_idx = np.argmin(lengths)
    best_route = routes[best_idx]
    deposit = Q / lengths[best_idx]
    for i in range(len(best_route)):
        a, b = best_route[i], best_route[(i + 1) % len(best_route)]
        pheromone[a, b] += deposit
        pheromone[b, a] += deposit

    return pheromone


def aco_step(pheromone, heuristic, distances, best_route, best_length):
    # Шаг 6: одна итерация ACO (повторяется заданное число раз)
    routes = construct_solutions(pheromone, heuristic, distances)
    lengths = calculate_all_lengths(routes, distances)
    pheromone = update_pheromones(pheromone, routes, lengths)

    # обновление глобального лучшего маршрута, если найден лучше текущего
    iteration_best_idx = np.argmin(lengths)
    if lengths[iteration_best_idx] < best_length:
        best_length = lengths[iteration_best_idx]
        best_route = routes[iteration_best_idx].copy()

    return pheromone, best_route, best_length