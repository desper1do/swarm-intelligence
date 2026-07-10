# demo.py
import time
from environment import make_env, get_env_info


def demo_episode(agent, delay=0.5):
    # прогоняет один эпизод обученного агента с визуализацией в реальном времени
    # delay - пауза между шагами в секундах, чтобы успевать смотреть.
    env = make_env(render_mode="human")
    state, info = env.reset()
    done = False
    total_reward = 0
    steps = 0

    # временно убираем исследование, хотим увидеть выученную политику,
    # а не случайные метания (agent.epsilon и так уже +-0.01 после обучения,
    # но обнулим явно для чистоты демонстрации)
    old_epsilon = agent.epsilon
    agent.epsilon = 0.0

    while not done:
        action = agent.choose_action(state)
        state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward
        steps += 1
        time.sleep(delay)

    agent.epsilon = old_epsilon  # возвращаем как было, вдруг агент еще понадобится
    env.close()

    result = "успех, дошёл до цели!" if total_reward > 0 else "провал, упал в лунку"
    print(f"Эпизод завершён за {steps} шагов — {result}")