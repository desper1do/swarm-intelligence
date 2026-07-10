# demo.py
import numpy as np
import torch
import torch.nn.functional as F
from pogema_env import make_env
from config import MAX_EPISODE_STEPS


def rollout_qmix(agent, device, seed, track_agent=0):
    # жадный прогон (epsilon=0) обученного QMIX для визуализации (п.4.1 задания):
    # траектория всех агентов + Q-значения выбранного агента на каждом шаге
    env = make_env(seed=seed)
    obs_list = env.reset(seed=seed)
    hidden = agent.init_hidden()

    positions = [env.get_agents_xy()]
    actions_hist = []
    q_values_hist = []

    old_epsilon = agent.epsilon
    agent.epsilon = 0.0
    for t in range(MAX_EPISODE_STEPS):
        obs = torch.as_tensor(np.stack(obs_list), dtype=torch.float32, device=device)
        with torch.no_grad():
            q_values, hidden = agent.agent_net(obs, agent._agent_id_onehot, hidden)
        actions = [int(q_values[i].argmax().item()) for i in range(agent.n_agents)]

        q_values_hist.append(q_values[track_agent].cpu().numpy())
        actions_hist.append(actions)

        obs_list, rewards, terminated, truncated, done, info = env.step(actions)
        positions.append(env.get_agents_xy())
        if all(done):
            break
    agent.epsilon = old_epsilon

    result = {
        "positions": positions,
        "actions": np.array(actions_hist),
        "q_values": np.array(q_values_hist),
        "obstacles": env.get_obstacles(),
        "targets": env.get_targets_xy(),
    }
    env.close()
    return result


def rollout_maddpg(agent, device, seed):
    # жадный прогон (без шума) обученного MADDPG для визуализации (п.4.2 задания)
    env = make_env(seed=seed)
    obs_list = env.reset(seed=seed)

    positions = [env.get_agents_xy()]
    actions_hist = []

    for t in range(MAX_EPISODE_STEPS):
        actions = agent.choose_actions(obs_list, explore=False)
        actions_hist.append(actions)
        obs_list, rewards, terminated, truncated, done, info = env.step(actions)
        positions.append(env.get_agents_xy())
        if all(done):
            break

    result = {
        "positions": positions,
        "actions": np.array(actions_hist),
        "obstacles": env.get_obstacles(),
        "targets": env.get_targets_xy(),
    }
    env.close()
    return result
