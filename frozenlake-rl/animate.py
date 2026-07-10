# animate.py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from config import ANIM_TARGET_FRAMES, MOVING_AVG_WINDOW, SEED
from environment import make_env, get_env_info
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


def animate_training(agent, run_episode, n_episodes, label="Q-Learning", color="blue"):
    # анимация - это и есть обучение (не отдельный черновой прогон): слева
    # Q-таблица (обновляется по ходу обучения), справа сходимость.
    # чтобы показать все n_episodes быстро, за один кадр анимации проходит не
    # один эпизод, а целая пачка (episodes_per_frame) - иначе на 10000 эпизодов
    # либо анимация идет вечность, либо обрывается на первых процентах обучения,
    # толком не успев показать, как агент выходит на сходимость
    episodes_per_frame = max(1, n_episodes // ANIM_TARGET_FRAMES)
    n_frames = n_episodes // episodes_per_frame

    env = make_env()
    state, info = env.reset(seed=SEED)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    im = ax1.imshow(agent.q_table, cmap="viridis", aspect="auto", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax1, label="Q-значение")
    ax1.set_xticks(range(len(ARROWS)))
    ax1.set_xticklabels([ARROWS[i] for i in range(len(ARROWS))])
    ax1.set_xlabel("Действие")
    ax1.set_ylabel("Состояние")

    rewards_history = []
    epsilon_history = []
    convergence_line, = ax2.plot([], [], color=color, label=label)
    ax2.set_xlim(0, n_episodes)
    ax2.set_ylim(-0.1, 1.1)
    ax2.set_xlabel("Эпизод")
    ax2.set_ylabel(f"Награда (скользящее среднее, окно={MOVING_AVG_WINDOW})")
    ax2.set_title("Сходимость")
    ax2.legend(loc="upper left")

    def update(frame):
        nonlocal state
        for _ in range(episodes_per_frame):
            total_reward = run_episode(env, agent, state)
            agent.decay_epsilon()
            state, info = env.reset()
            rewards_history.append(total_reward)
            epsilon_history.append(agent.epsilon)

        # тепловая карта перерисовывается с учетом текущего разброса Q-значений
        im.set_data(agent.q_table)
        im.set_clim(agent.q_table.min(), agent.q_table.max())

        window = min(len(rewards_history), MOVING_AVG_WINDOW)
        smoothed = np.convolve(rewards_history, np.ones(window) / window, mode="valid")
        convergence_line.set_data(range(len(smoothed)), smoothed)

        episode_no = len(rewards_history)
        ax1.set_title(f"{label}: Q-таблица | эпизод {episode_no}/{n_episodes} | epsilon={agent.epsilon:.3f}")
        return im, convergence_line

    anim = animation.FuncAnimation(fig, update, frames=n_frames, interval=30, repeat=False)
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
    return rewards_history, epsilon_history
