# animate.py
import time
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from config import ANIM_TARGET_FRAMES, MOVING_AVG_WINDOW, QMIX, MADDPG, MAX_EPISODE_STEPS
from pogema_env import make_env
from agent_qmix import QMIXAgent
from agent_maddpg import MADDPGAgent
from buffer_qmix import EpisodeBuffer
from buffer_maddpg import ReplayBuffer
from train_qmix import collect_episode as collect_episode_qmix, train_step as train_step_qmix
from train_maddpg import train_step as train_step_maddpg
from demo import rollout_qmix, rollout_maddpg
from visualize import AGENT_COLORS


def _draw_trajectories(ax, rollout, n_agents):
    # снимок карты POGEMA с траекторией текущей жадной политики - переиспользуем
    # ту же раскладку, что и в visualize.plot_trajectory, но перерисовываем на
    # уже существующих осях (для анимации), а не создаем новую фигуру каждый раз
    ax.clear()
    obstacles = rollout["obstacles"]
    positions = np.array(rollout["positions"])
    targets = rollout["targets"]
    rows, cols = obstacles.shape

    ax.imshow(obstacles, cmap="Greys", origin="upper", alpha=0.6)
    for i in range(n_agents):
        color = AGENT_COLORS[i % len(AGENT_COLORS)]
        path = positions[:, i, :]
        ax.plot(path[:, 1], path[:, 0], color=color, linewidth=2, marker="o", markersize=3)
        ax.plot(path[0, 1], path[0, 0], color=color, marker="s", markersize=10, markeredgecolor="black")
        tx, ty = targets[i]
        ax.plot(ty, tx, color=color, marker="*", markersize=14, markeredgecolor="black")
    ax.set_xlim(-0.5, cols - 0.5)
    ax.set_ylim(rows - 0.5, -0.5)
    ax.set_xticks([])
    ax.set_yticks([])


def _replay_evolution(obstacles, targets, snapshots, n_agents, label):
    # зацикленный повтор эволюции траектории по ходу обучения: снимки жадной
    # политики, накопленные во время основной анимации (та же картинка, что
    # обновлялась в левой панели каждый кадр), проигрываются один за другим от
    # ранних эпизодов к поздним, затем повтор с начала (repeat=True) - наглядно
    # видно, как менялось поведение агентов, включая ухудшение, если оно было
    fig, ax = plt.subplots(figsize=(6, 6))

    def update(i):
        snap = snapshots[i]
        rollout = {"obstacles": obstacles, "positions": snap["positions"], "targets": targets}
        _draw_trajectories(ax, rollout, n_agents)
        ax.set_title(f"{label}: эволюция траектории | эпизод {snap['episode']}")
        return ax,

    anim = animation.FuncAnimation(fig, update, frames=len(snapshots), interval=200, repeat=True)
    plt.tight_layout()
    plt.show()


def _update_best(agent, success_history, best, warmup):
    # best-checkpoint: как только сглаженный success_rate ставит новый рекорд,
    # запоминаем веса. warmup не даёт "поймать" случайно удачный ранний эпизод.
    # в конце обучения по этим весам строятся финальные графики/демо
    ep_no = len(success_history)
    if ep_no < warmup:
        return
    score = float(np.mean(success_history[-MOVING_AVG_WINDOW:]))
    if score > best["score"]:
        best["score"] = score
        best["episode"] = ep_no
        best["snap"] = agent.snapshot()


def _run_animation(fig, ax1, ax2, n_episodes, episodes_per_frame, update_fn, label):
    n_frames = max(1, n_episodes // episodes_per_frame)
    anim = animation.FuncAnimation(fig, update_fn, frames=n_frames, interval=30, repeat=False)
    anim.running = True

    # пауза по пробелу (тот же прием, что и в остальных практических)
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


def animate_qmix(n_episodes, device, seed, label="QMIX", color="#2c5282"):
    # анимация - это и есть обучение QMIX (шаги 2-10 плана), не отдельный
    # черновой прогон: слева траектория текущей жадной политики на карте
    # POGEMA (снимок раз в кадр), справа сходимость по всей истории обучения
    env = make_env(seed=seed)
    env.reset(seed=seed)
    state_dim = env.get_global_state().shape[0]

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    agent = QMIXAgent(env.obs_shape, env.n_actions, env.n_agents, state_dim, device)
    buffer = EpisodeBuffer(QMIX.BUFFER_SIZE, MAX_EPISODE_STEPS, env.n_agents, env.obs_shape, state_dim)

    episodes_per_frame = max(1, n_episodes // ANIM_TARGET_FRAMES)
    history = {"reward": [], "epsilon": [], "loss": [], "q_tot": [], "collisions": [], "success_rate": []}
    snapshots = []
    best = {"score": -np.inf, "snap": None, "episode": 0}
    warmup = min(MOVING_AVG_WINDOW * 3, n_episodes // 3)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.5))
    convergence_line, = ax2.plot([], [], color=color, label=label)
    ax2.set_xlim(0, n_episodes)
    ax2.set_xlabel("Номер эпизода")
    ax2.set_ylabel(f"Суммарная награда (скользящее среднее, окно={MOVING_AVG_WINDOW})")
    ax2.set_title("Сходимость")
    ax2.legend(loc="upper left")

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    t0 = time.time()

    def update(frame):
        for _ in range(episodes_per_frame):
            ep = len(history["reward"])
            r, collisions, success_rate, length = collect_episode_qmix(
                env, agent, buffer, seed=seed if ep == 0 else None
            )

            loss, q_tot = np.nan, np.nan
            if len(buffer) >= 4:
                result = train_step_qmix(agent, buffer, QMIX.BATCH_SIZE, device)
                if result is not None:
                    loss, q_tot = result
                    agent.soft_update()
            agent.decay_epsilon()

            history["reward"].append(r)
            history["epsilon"].append(agent.epsilon)
            history["loss"].append(loss)
            history["q_tot"].append(q_tot)
            history["collisions"].append(collisions)
            history["success_rate"].append(success_rate)

        _update_best(agent, history["success_rate"], best, warmup)
        rollout = rollout_qmix(agent, device, seed)
        _draw_trajectories(ax1, rollout, env.n_agents)
        ep_no = len(history["reward"])
        ax1.set_title(f"{label}: жадная политика | эпизод {ep_no}/{n_episodes} | epsilon={agent.epsilon:.3f}")
        snapshots.append({"episode": ep_no, "positions": rollout["positions"]})

        rewards = history["reward"]
        window = min(len(rewards), MOVING_AVG_WINDOW)
        smoothed = np.convolve(rewards, np.ones(window) / window, mode="valid")
        convergence_line.set_data(range(len(smoothed)), smoothed)
        ax2.relim()
        ax2.autoscale_view()
        return ax1, convergence_line

    _run_animation(fig, ax1, ax2, n_episodes, episodes_per_frame, update, label)

    history["train_time"] = time.time() - t0
    history["peak_memory_mb"] = (
        torch.cuda.max_memory_allocated(device) / 1e6 if device.type == "cuda" else 0.0
    )

    # для финальных графиков/демо возвращаем ЛУЧШИЕ веса, а не последние
    if best["snap"] is not None:
        agent.load_snapshot(best["snap"])
        print(f"{label}: восстановлены лучшие веса (эпизод {best['episode']}, success_rate={best['score']:.3f})")

    print(f"Повтор эволюции траектории {label} (закройте окно, чтобы продолжить)")
    # обстановку для повтора берем со свежей карты с тем же SEED (та же, что
    # использовалась во всех снимках rollout_qmix) - training env к этому моменту
    # уже "уехал" на случайную карту последнего эпизода (env.reset() без seed)
    replay_env = make_env(seed=seed)
    replay_env.reset(seed=seed)
    _replay_evolution(replay_env.get_obstacles(), replay_env.get_targets_xy(), snapshots, env.n_agents, label)
    replay_env.close()

    env.close()
    return agent, history


def animate_maddpg(n_episodes, device, seed, label="MADDPG", color="#2d6a4f"):
    # анимация - это и есть обучение MADDPG (шаги 2-10 плана): обучение идет
    # раз в MADDPG.UPDATE_EVERY шагов среды (не раз в эпизод), снимок политики
    # на карте обновляется раз в кадр анимации
    env = make_env(seed=seed)
    env.reset(seed=seed)

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    agent = MADDPGAgent(env.obs_shape, env.n_actions, env.n_agents, device)
    buffer = ReplayBuffer(MADDPG.BUFFER_SIZE)

    episodes_per_frame = max(1, n_episodes // ANIM_TARGET_FRAMES)
    history = {"reward": [], "sigma": [], "critic_loss": [], "actor_loss": [], "collisions": [], "success_rate": []}
    snapshots = []
    best = {"score": -np.inf, "snap": None, "episode": 0}
    warmup = min(MOVING_AVG_WINDOW * 2, n_episodes // 3)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.5))
    convergence_line, = ax2.plot([], [], color=color, label=label)
    ax2.set_xlim(0, n_episodes)
    ax2.set_xlabel("Номер эпизода")
    ax2.set_ylabel("Суммарная награда (скользящее среднее)")
    ax2.set_title("Сходимость")
    ax2.legend(loc="upper left")

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    t0 = time.time()
    global_step = 0

    def update(frame):
        nonlocal global_step
        for _ in range(episodes_per_frame):
            ep = len(history["reward"])
            obs_list = env.reset(seed=seed if ep == 0 else None)
            total_reward = np.zeros(agent.n_agents)
            total_collisions = 0
            final_terminated = [False] * agent.n_agents
            ep_critic_losses, ep_actor_losses = [], []

            for t in range(MAX_EPISODE_STEPS):
                actions = agent.choose_actions(obs_list, explore=True)
                next_obs_list, rewards, terminated, truncated, done, info = env.step(actions)
                buffer.add(obs_list, actions, rewards, next_obs_list, done)
                total_reward += np.array(rewards)
                total_collisions += info["collisions"]
                final_terminated = terminated
                obs_list = next_obs_list
                global_step += 1

                if global_step % MADDPG.UPDATE_EVERY == 0 and len(buffer) >= MADDPG.BATCH_SIZE:
                    result = train_step_maddpg(agent, buffer, MADDPG.BATCH_SIZE, device)
                    if result is not None:
                        ep_critic_losses.append(result[0])
                        if not np.isnan(result[1]):
                            ep_actor_losses.append(result[1])
                if all(done):
                    break

            agent.decay_sigma()
            history["reward"].append(total_reward.sum())
            history["sigma"].append(agent.sigma)
            history["critic_loss"].append(np.mean(ep_critic_losses) if ep_critic_losses else np.nan)
            history["actor_loss"].append(np.mean(ep_actor_losses) if ep_actor_losses else np.nan)
            history["collisions"].append(total_collisions)
            history["success_rate"].append(float(np.mean(final_terminated)))

        _update_best(agent, history["success_rate"], best, warmup)
        rollout = rollout_maddpg(agent, device, seed)
        _draw_trajectories(ax1, rollout, env.n_agents)
        ep_no = len(history["reward"])
        ax1.set_title(f"{label}: жадная политика | эпизод {ep_no}/{n_episodes} | sigma={agent.sigma:.3f}")
        snapshots.append({"episode": ep_no, "positions": rollout["positions"]})

        rewards = history["reward"]
        window = min(len(rewards), MOVING_AVG_WINDOW)
        smoothed = np.convolve(rewards, np.ones(window) / window, mode="valid")
        convergence_line.set_data(range(len(smoothed)), smoothed)
        ax2.relim()
        ax2.autoscale_view()
        return ax1, convergence_line

    _run_animation(fig, ax1, ax2, n_episodes, episodes_per_frame, update, label)

    history["train_time"] = time.time() - t0
    history["peak_memory_mb"] = (
        torch.cuda.max_memory_allocated(device) / 1e6 if device.type == "cuda" else 0.0
    )

    # для финальных графиков/демо возвращаем ЛУЧШИЕ веса, а не последние
    if best["snap"] is not None:
        agent.load_snapshot(best["snap"])
        print(f"{label}: восстановлены лучшие веса (эпизод {best['episode']}, success_rate={best['score']:.3f})")

    print(f"Повтор эволюции траектории {label} (закройте окно, чтобы продолжить)")
    replay_env = make_env(seed=seed)
    replay_env.reset(seed=seed)
    _replay_evolution(replay_env.get_obstacles(), replay_env.get_targets_xy(), snapshots, env.n_agents, label)
    replay_env.close()

    env.close()
    return agent, history
