# animate.py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from config import ANIM_EPISODES, MOVING_AVG_WINDOW, SEED
from environment import make_env
from visualize import ARROWS


def run_episode_qlearning(env, agent, state):
    # один эпизод Q-Learning: обычный шаг агента + обновление по max Q(s',a')
    done = False
    total_reward = 0
    while not done:
        action = agent.choose_action(state)
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        agent.update(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward
    return total_reward


def run_episode_sarsa(env, agent, state):
    # один эпизод SARSA: действие выбирается заранее и переносится в след. шаг (a')
    done = False
    total_reward = 0
    action = agent.choose_action(state)
    while not done:
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        next_action = agent.choose_action(next_state)
        agent.update(state, action, reward, next_state, next_action, done)
        state, action = next_state, next_action
        total_reward += reward
    return total_reward


def animate_training(agent, run_episode, label="Q-Learning", color="blue"):
    # анимация обучения в реальном времени: слева Q-таблица (обновляется по ходу
    # обучения), справа сходимость. n_episodes ограничено ANIM_EPISODES,
    # иначе анимация растянется на то же время, что и полное обучение (10000 эпизодов)
    env = make_env()
    state, info = env.reset(seed=SEED)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # награды в CliffWalking отрицательные (-1 за шаг, -100 за обрыв), поэтому
    # диапазон цветовой шкалы берем [-100, 0], а не [0, 1] как во FrozenLake
    im = ax1.imshow(agent.q_table, cmap="viridis", aspect="auto", vmin=-100, vmax=0)
    plt.colorbar(im, ax=ax1, label="Q-значение")
    ax1.set_xticks(range(len(ARROWS)))
    ax1.set_xticklabels([ARROWS[i] for i in range(len(ARROWS))])
    ax1.set_xlabel("Действие")
    ax1.set_ylabel("Состояние")

    convergence_history = []
    convergence_line, = ax2.plot([], [], color=color, label=label)
    ax2.set_xlim(0, ANIM_EPISODES)
    ax2.set_xlabel("Эпизод")
    ax2.set_ylabel(f"Награда (скользящее среднее, окно={MOVING_AVG_WINDOW})")
    ax2.set_title("Сходимость")
    ax2.legend(loc="upper left")

    def update(frame):
        nonlocal state
        total_reward = run_episode(env, agent, state)
        agent.decay_epsilon()
        state, info = env.reset()

        # тепловая карта перерисовывается с учетом текущего разброса Q-значений
        im.set_data(agent.q_table)
        im.set_clim(agent.q_table.min(), agent.q_table.max())

        convergence_history.append(total_reward)
        smoothed = np.convolve(
            convergence_history,
            np.ones(min(len(convergence_history), MOVING_AVG_WINDOW)) / min(len(convergence_history), MOVING_AVG_WINDOW),
            mode="valid",
        )
        convergence_line.set_data(range(len(smoothed)), smoothed)
        # диапазон наград на CliffWalking заранее не известен (в отличие от 0/1
        # во FrozenLake), поэтому масштаб оси Y пересчитываем на каждом кадре
        ax2.relim()
        ax2.autoscale_view()

        ax1.set_title(f"{label}: Q-таблица | эпизод {frame + 1} | epsilon={agent.epsilon:.2f}")
        return im, convergence_line

    anim = animation.FuncAnimation(fig, update, frames=ANIM_EPISODES, interval=50, repeat=False)
    anim.running = True

    # пауза по пробелу (тот же прием, что и в анимациях ACO/PSO)
    def on_key(event):
        if event.key == " ":
            if anim.running:
                anim.event_source.stop()
            else:
                anim.event_source.start()
            anim.running = not anim.running

    fig.canvas.mpl_connect("key_press_event", on_key)
    plt.tight_layout()
    plt.show()
    env.close()
