# train_qmix.py
import time
import random
import numpy as np
import torch
from config import QMIX, MAX_EPISODE_STEPS, MOVING_AVG_WINDOW
from pogema_env import make_env
from agent_qmix import QMIXAgent
from buffer_qmix import EpisodeBuffer


def collect_episode(env, agent, buffer, seed=None):
    # шаги 2-4 плана QMIX: сбор одного эпизода в буфер целиком
    episode = buffer.start_episode()
    obs_list = env.reset(seed=seed)
    hidden = agent.init_hidden()

    total_reward = 0.0
    total_collisions = 0
    final_terminated = [False] * agent.n_agents
    T = buffer.max_episode_steps

    for t in range(T):
        state = env.get_global_state()
        actions, hidden = agent.choose_actions(obs_list, hidden)
        next_obs_list, rewards, terminated, truncated, done, info = env.step(actions)

        team_reward = float(sum(rewards))
        episode_done = float(all(done))

        episode["obs"][t] = np.stack(obs_list)
        episode["state"][t] = state
        episode["actions"][t] = actions
        episode["team_reward"][t] = team_reward
        episode["done"][t] = episode_done
        episode["filled"][t] = 1.0

        total_reward += team_reward
        total_collisions += info["collisions"]
        final_terminated = terminated
        obs_list = next_obs_list

        if all(done):
            break

    buffer.add_episode(episode)
    # success_rate - доля агентов, дошедших до цели к концу эпизода. Награда
    # сильно зашумлена штрафом за столкновения, поэтому success_rate - более
    # честный показатель того, решает ли алгоритм саму задачу поиска пути
    success_rate = float(np.mean(final_terminated))
    return total_reward, total_collisions, success_rate, t + 1


def train_step(agent, buffer, batch_size, device):
    # шаги 5-7 плана QMIX: батч эпизодов -> Q_tot по всей траектории через BPTT
    # (GRU разворачивается по времени и для основной, и для целевой сети) -> MSE(y, Q_tot)
    if len(buffer) == 0:
        return None

    batch = buffer.sample(batch_size)
    B, T, n_agents = batch["actions"].shape

    obs = torch.as_tensor(batch["obs"], dtype=torch.float32, device=device)
    state = torch.as_tensor(batch["state"], dtype=torch.float32, device=device)
    actions = torch.as_tensor(batch["actions"], dtype=torch.int64, device=device)
    team_reward = torch.as_tensor(batch["team_reward"], dtype=torch.float32, device=device)
    done = torch.as_tensor(batch["done"], dtype=torch.float32, device=device)
    filled = torch.as_tensor(batch["filled"], dtype=torch.float32, device=device)

    agent_id = agent._agent_id_onehot.unsqueeze(0).expand(B, -1, -1).reshape(B * n_agents, n_agents)

    hidden = agent.agent_net.init_hidden(B * n_agents, device)
    target_hidden = agent.target_agent_net.init_hidden(B * n_agents, device)

    q_tot_list, q_tot_target_list = [], []
    for t in range(T):
        obs_t = obs[:, t].reshape(B * n_agents, *obs.shape[3:])

        q_vals, hidden = agent.agent_net(obs_t, agent_id, hidden)
        q_vals = q_vals.view(B, n_agents, -1)
        chosen_q = q_vals.gather(-1, actions[:, t].unsqueeze(-1)).squeeze(-1)  # (B, n_agents)
        q_tot_list.append(agent.mixer(chosen_q, state[:, t]))

        with torch.no_grad():
            target_q_vals, target_hidden = agent.target_agent_net(obs_t, agent_id, target_hidden)
            target_q_vals = target_q_vals.view(B, n_agents, -1)
            # Double-Q: действие выбирает ОНЛАЙН-сеть (argmax), а оценивает его
            # TARGET-сеть - это снижает переоценку Q-значений (overestimation bias),
            # из-за которой ванильный max по target-сети раскачивает обучение на
            # длинной дистанции (награда растет до пика, а потом деградирует)
            online_argmax = q_vals.detach().argmax(dim=-1, keepdim=True)  # (B, n_agents, 1)
            double_q = target_q_vals.gather(-1, online_argmax).squeeze(-1)  # (B, n_agents)
            q_tot_target_list.append(agent.target_mixer(double_q, state[:, t]))

    q_tot = torch.stack(q_tot_list, dim=1)  # (B, T)
    q_tot_target = torch.stack(q_tot_target_list, dim=1)  # (B, T)

    # цель шага t использует Q_tot_target следующего шага (t+1), последний шаг
    # эпизода бутстрапа не получает (done=1 обнуляет второе слагаемое)
    next_q_tot_target = torch.cat([q_tot_target[:, 1:], torch.zeros(B, 1, device=device)], dim=1)
    y = team_reward + QMIX.GAMMA * (1 - done) * next_q_tot_target

    td_error = (q_tot - y.detach()) * filled
    loss = (td_error ** 2).sum() / filled.sum().clamp(min=1.0)

    agent.optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(
        list(agent.agent_net.parameters()) + list(agent.mixer.parameters()), max_norm=10.0
    )
    agent.optimizer.step()

    return loss.item(), q_tot.mean().item()


def run_training(n_episodes, device, seed, verbose=True):
    # шаг 10 плана QMIX: полный цикл обучения, собирает историю метрик для визуализации
    env = make_env(seed=seed)
    env.reset(seed=seed)
    state_dim = env.get_global_state().shape[0]

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    agent = QMIXAgent(env.obs_shape, env.n_actions, env.n_agents, state_dim, device)
    buffer = EpisodeBuffer(QMIX.BUFFER_SIZE, MAX_EPISODE_STEPS, env.n_agents, env.obs_shape, state_dim)

    history = {"reward": [], "epsilon": [], "loss": [], "q_tot": [], "collisions": [], "success_rate": []}
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    t0 = time.time()

    for ep in range(n_episodes):
        r, collisions, success_rate, length = collect_episode(env, agent, buffer, seed=seed if ep == 0 else None)

        loss, q_tot = (np.nan, np.nan)
        if len(buffer) >= 4:
            result = train_step(agent, buffer, QMIX.BATCH_SIZE, device)
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

        if verbose and (ep + 1) % 100 == 0:
            avg_r = np.mean(history["reward"][-MOVING_AVG_WINDOW:])
            print(f"QMIX ep {ep + 1}/{n_episodes}: avg_reward={avg_r:.2f} epsilon={agent.epsilon:.3f}")

    history["train_time"] = time.time() - t0
    history["peak_memory_mb"] = (
        torch.cuda.max_memory_allocated(device) / 1e6 if device.type == "cuda" else 0.0
    )
    env.close()
    return agent, history
