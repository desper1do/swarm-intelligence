from pso import initialize_swarm, update_swarm
from config import MAX_ITERATIONS
from functions import rastrigin, himmelblau
from visualize import animate_pso

# выбор целевой функции для оптимизации (rastrigin/himmelblau)
objective_function = rastrigin

if __name__ == "__main__":
    # анимация процесса оптимизации
    animate_pso(objective_function)

    # инициализация роя и оптимизация
    position, velocity, pbest_position, pbest_value, gbest_position, gbest_value = initialize_swarm(objective_function)
    
    for iteration in range(MAX_ITERATIONS):
        position, velocity, pbest_position, pbest_value, gbest_position, gbest_value = update_swarm(
            position, velocity, pbest_position, pbest_value, gbest_position, gbest_value, objective_function
        )
        if iteration % 10 == 0:
            print(f"Итерация {iteration}: gbest_value = {gbest_value:.10f}")
    
    print("gbest_value:", gbest_value)
    print("gbest_position:", gbest_position)
