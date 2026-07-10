# train_maddpg.py
import time
import random
import numpy as np
import torch
import torch.nn.functional as F
from config import MADDPG, MAX_EPISODE_STEPS, MOVING_AVG_WINDOW
from pogema_env import make_env
from agent_maddpg import MADDPGAgent
from buffer_maddpg import ReplayBuffer


def run_episode(env, agent, buffer, device, global_step, seed=None):
    # шаги 2-5 плана MADDPG: собрать эпизод переходов в обычный буфер (не эпизодический,
    # т.к. в MADDPG нет RNN - каждый переход самостоятелен), обучение запускается
    # не раз в эпизод, а каждые MADDPG.UPDATE_EVERY шагов среды (п.3.2.2 задания)
    obs_list = env.reset(seed=seed)
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
            result = train_step(agent, buffer, MADDPG.BATCH_SIZE, device)
            if result is not None:
                ep_critic_losses.append(result[0])
                if not np.isnan(result[1]):
                    ep_actor_losses.append(result[1])

        if all(done):
            break

    critic_loss = float(np.mean(ep_critic_losses)) if ep_critic_losses else np.nan
    actor_loss = float(np.mean(ep_actor_losses)) if ep_actor_losses else np.nan
    # success_rate - доля агентов, дошедших до цели (награда сильно зашумлена
    # штрафом за столкновения, поэтому это более честный индикатор решения задачи)
    success_rate = float(np.mean(final_terminated))
    return total_reward.sum(), total_collisions, success_rate, t + 1, global_step, critic_loss, actor_loss


def train_step(agent, buffer, batch_size, device):
    # шаги 6-9 плана MADDPG с TD3-стабилизацией: критики (twin) по MSE к цели
    # y = r + gamma*min(Q1',Q2'), актеры - по градиенту политики через
    # дифференцируемую дискретизацию (Gumbel-Softmax). Актер и целевые сети
    # обновляются реже критика (delayed policy update, раз в POLICY_DELAY шагов) -
    # и soft_update тоже вызывается внутри, только на отложенных шагах
    if len(buffer) < batch_size:
        return None

    agent._update_count += 1
    update_actor = (agent._update_count % MADDPG.POLICY_DELAY == 0)

    obs, actions, rewards, next_obs, done = buffer.sample(batch_size)
    obs = torch.as_tensor(obs, dtype=torch.float32, device=device)
    actions = torch.as_tensor(actions, dtype=torch.int64, device=device)
    rewards = torch.as_tensor(rewards, dtype=torch.float32, device=device)
    next_obs = torch.as_tensor(next_obs, dtype=torch.float32, device=device)
    done = torch.as_tensor(done, dtype=torch.float32, device=device)

    n_agents = agent.n_agents
    actions_onehot = F.one_hot(actions, agent.n_actions).float()  # (B, n, n_actions)

    with torch.no_grad():
        # target policy smoothing: к логитам целевой политики добавляем шум,
        # чтобы критик не переобучался на резкий argmax (TD3)
        target_action_list = []
        for i in range(n_agents):
            logits = agent.target_actors[i](next_obs[:, i])
            noise = torch.clamp(torch.randn_like(logits) * MADDPG.TARGET_NOISE,
                                -MADDPG.NOISE_CLIP, MADDPG.NOISE_CLIP)
            target_action_list.append(F.gumbel_softmax(logits + noise, hard=True))
        target_actions_onehot = torch.stack(target_action_list, dim=1)

    critic_losses, actor_losses = [], []

    for i in range(n_agents):
        with torch.no_grad():
            tq1, tq2 = agent.target_critics[i](next_obs, target_actions_onehot)
            target_q = torch.min(tq1, tq2)  # min гасит переоценку (TD3)
            y = rewards[:, i] + MADDPG.GAMMA * (1 - done[:, i]) * target_q
        q1, q2 = agent.critics[i](obs, actions_onehot)
        critic_loss = F.mse_loss(q1, y) + F.mse_loss(q2, y)

        agent.critic_optimizers[i].zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.critics[i].parameters(), max_norm=10.0)
        agent.critic_optimizers[i].step()
        critic_losses.append(critic_loss.item())

    if not update_actor:
        # отложенное обновление: на этом шаге учим только критиков (без актера,
        # без soft_update целевых сетей)
        return float(np.mean(critic_losses)), np.nan

    for i in range(n_agents):
        # действие агента i заменяем на дифференцируемое (Gumbel-Softmax), действия
        # остальных агентов берем как есть из буфера - см. формулу градиента политики (п.3.1)
        oh_i = F.gumbel_softmax(agent.actors[i](obs[:, i]), hard=True)
        current_actions_onehot = torch.cat(
            [actions_onehot[:, :i], oh_i.unsqueeze(1), actions_onehot[:, i + 1:]], dim=1
        )
        actor_loss = -agent.critics[i].q1(obs, current_actions_onehot).mean()

        agent.actor_optimizers[i].zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.actors[i].parameters(), max_norm=10.0)
        agent.actor_optimizers[i].step()
        actor_losses.append(actor_loss.item())

    agent.soft_update()  # целевые сети обновляем на том же (отложенном) шаге, что и актера
    return float(np.mean(critic_losses)), float(np.mean(actor_losses))


def run_training(n_episodes, device, seed, verbose=True):
    # шаг 10 плана MADDPG: полный цикл обучения, собирает историю метрик для визуализации
    env = make_env(seed=seed)
    env.reset(seed=seed)

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    agent = MADDPGAgent(env.obs_shape, env.n_actions, env.n_agents, device)
    buffer = ReplayBuffer(MADDPG.BUFFER_SIZE)

    history = {"reward": [], "sigma": [], "critic_loss": [], "actor_loss": [], "collisions": [], "success_rate": []}
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    t0 = time.time()
    global_step = 0

    for ep in range(n_episodes):
        r, collisions, success_rate, length, global_step, critic_loss, actor_loss = run_episode(
            env, agent, buffer, device, global_step, seed=seed if ep == 0 else None
        )
        agent.decay_sigma()

        history["reward"].append(r)
        history["sigma"].append(agent.sigma)
        history["critic_loss"].append(critic_loss)
        history["actor_loss"].append(actor_loss)
        history["collisions"].append(collisions)
        history["success_rate"].append(success_rate)

        if verbose and (ep + 1) % 20 == 0:
            avg_r = np.mean(history["reward"][-MOVING_AVG_WINDOW:])
            print(f"MADDPG ep {ep + 1}/{n_episodes}: avg_reward={avg_r:.2f} sigma={agent.sigma:.3f}")

    history["train_time"] = time.time() - t0
    history["peak_memory_mb"] = (
        torch.cuda.max_memory_allocated(device) / 1e6 if device.type == "cuda" else 0.0
    )
    env.close()
    return agent, history
