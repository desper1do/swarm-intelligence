import matplotlib.pyplot as plt
import matplotlib.animation as animation
from config import MAX_ITERATIONS
from tsp_graph import generate_cities, compute_distance_matrix
from aco import initialize_aco, aco_step


def animate_aco():
    cities = generate_cities()
    distances = compute_distance_matrix(cities)
    pheromone, heuristic, best_route, best_length = initialize_aco(distances)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.scatter(cities[:, 0], cities[:, 1], c='black', s=50, zorder=3)

    # линии феромона рисуем один раз, прозрачность будем менять по кадрам
    n = len(cities)
    pheromone_lines = []
    for i in range(n):
        for j in range(i + 1, n):
            line, = ax1.plot(
                [cities[i, 0], cities[j, 0]], [cities[i, 1], cities[j, 1]],
                color='blue', alpha=0.0, linewidth=1, zorder=1
            )
            pheromone_lines.append((i, j, line))

    route_line, = ax1.plot([], [], color='red', linewidth=2, zorder=2, label='лучший маршрут')
    ax1.legend(loc='upper right')

    convergence_history = []
    convergence_line, = ax2.plot([], [], color='green')
    ax2.set_xlim(0, MAX_ITERATIONS)
    ax2.set_xlabel('Итерация')
    ax2.set_ylabel('Длина маршрута')
    ax2.set_title('Сходимость')

    def update(frame):
        nonlocal pheromone, best_route, best_length
        pheromone, best_route, best_length = aco_step(pheromone, heuristic, distances, best_route, best_length)

        # толщина и прозрачность линий зависят от относительной концентрации феромона
        max_pheromone = pheromone.max()
        for i, j, line in pheromone_lines:
            intensity = pheromone[i, j] / max_pheromone
            line.set_alpha(min(intensity, 1.0))
            line.set_linewidth(1 + 3 * intensity)

        route_coords = cities[list(best_route) + [best_route[0]]]  # замыкаем маршрут визуально
        route_line.set_data(route_coords[:, 0], route_coords[:, 1])

        convergence_history.append(best_length)
        convergence_line.set_data(range(len(convergence_history)), convergence_history)
        ax2.set_ylim(0, max(convergence_history) * 1.1)

        ax1.set_title(f"Итерация {frame} | длина маршрута = {best_length:.2f}")
        return route_line, convergence_line

    anim = animation.FuncAnimation(fig, update, frames=MAX_ITERATIONS, interval=150, repeat=False)
    anim.running = True

    def on_key(event):
        if event.key == ' ':
            if anim.running:
                anim.event_source.stop()
            else:
                anim.event_source.start()
            anim.running = not anim.running

    fig.canvas.mpl_connect('key_press_event', on_key)
    plt.tight_layout()
    plt.show()

