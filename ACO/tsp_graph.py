import numpy as np
from config import N_CITIES, SEARCH_MIN, SEARCH_MAX, RANDOM_SEED

def generate_cities():
    # случайные координаты городов на плоскости (фиксированный seed для воспроизводимости)
    rng = np.random.default_rng(RANDOM_SEED)
    return rng.uniform(SEARCH_MIN, SEARCH_MAX, size=(N_CITIES, 2))

def compute_distance_matrix(cities):
    # матрица евклидовых расстояний между городами, на диагонали inf (город сам с собой не соединен)
    n = len(cities)
    distances = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            dx = cities[i, 0] - cities[j, 0]
            dy = cities[i, 1] - cities[j, 1]
            distances[i, j] = np.sqrt(dx**2 + dy**2)
    np.fill_diagonal(distances, np.inf)
    return distances