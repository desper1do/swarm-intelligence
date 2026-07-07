from visualize import animate_aco
from tsp_graph import generate_cities, compute_distance_matrix
from aco import initialize_aco, aco_step
from config import MAX_ITERATIONS

if __name__ == "__main__":
    # анимация процесса оптимизации
    animate_aco()

    cities = generate_cities()
    distances = compute_distance_matrix(cities)
    # инициализация феромонов и оптимизация маршрута
    pheromone, heuristic, best_route, best_length = initialize_aco(distances)

    for iteration in range(MAX_ITERATIONS):
        pheromone, best_route, best_length = aco_step(pheromone, heuristic, distances, best_route, best_length)
        if iteration % 10 == 0:
            print(f"Итерация {iteration}: длина = {best_length:.2f}")

    print("best_route:", best_route)
    print("best_length:", best_length)