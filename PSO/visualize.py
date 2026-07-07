import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from config import SEARCH_MIN, SEARCH_MAX, MAX_ITERATIONS
from pso import initialize_swarm, update_swarm


def animate_pso(objective_function):
    # инициализация роя
    position, velocity, pbest_position, pbest_value, gbest_position, gbest_value = initialize_swarm(objective_function)

    # фигура с двумя графиками (слева рой, справа сходимость)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,5))

    # контур функции
    x = np.linspace(SEARCH_MIN, SEARCH_MAX, 200)
    y = np.linspace(SEARCH_MIN, SEARCH_MAX, 200)
    X, Y = np.meshgrid(x,y)
    Z = objective_function(X,Y)
    ax1.contourf(X, Y, Z, levels=50, cmap='viridis')
    ax1.set_xlim(SEARCH_MIN, SEARCH_MAX)
    ax1.set_ylim(SEARCH_MIN, SEARCH_MAX)
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')

    # точки роя: текущая позиция, личный лучший (pbest) каждой частицы и глобальный лучший (gbest)
    particles_plot = ax1.scatter(position[:, 0], position[:, 1], c='white', edgecolors='black', label='текущая позиция')
    pbest_plot = ax1.scatter(pbest_position[:, 0], pbest_position[:, 1], c='none', edgecolors='lime', marker='o', label='pbest')
    gbest_plot = ax1.scatter([gbest_position[0]], [gbest_position[1]], c='red', marker='*', s=300, label='gbest')
    ax1.legend(loc='upper right')
    title = ax1.set_title('Итерация 0')

    # график сходимости
    convergence_history = [gbest_value]
    convergence_line, = ax2.plot(convergence_history, color='blue')
    ax2.set_xlim(0, MAX_ITERATIONS)
    ax2.set_xlabel('Итерация')
    ax2.set_ylabel('gbest_value')
    ax2.set_title('Сходимость')

    def update(frame):
        nonlocal position, velocity, pbest_position, pbest_value, gbest_position, gbest_value

        position, velocity, pbest_position, pbest_value, gbest_position, gbest_value = update_swarm(
            position, velocity, pbest_position, pbest_value, gbest_position, gbest_value, objective_function
        )

        particles_plot.set_offsets(position)
        pbest_plot.set_offsets(pbest_position)
        gbest_plot.set_offsets([gbest_position])
        title.set_text(f"{objective_function.__name__} | Итерация {frame} | gbest = {gbest_value:.10f}")

        convergence_history.append(gbest_value)
        convergence_line.set_data(range(len(convergence_history)), convergence_history)
        ax2.set_ylim(0, max(convergence_history) * 1.1)

        return particles_plot, pbest_plot, gbest_plot, title, convergence_line

    # пауза по пробелу
    anim = animation.FuncAnimation(fig, update, frames=MAX_ITERATIONS, interval=100, repeat=False)
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
