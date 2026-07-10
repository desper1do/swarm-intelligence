# demo.py
import time
from environment import make_env


def demo_episode(agent, delay=0.5, max_steps=200):
    # прогоняет один эпизод обученного агента с визуализацией в реальном времени.
    # max_steps - защита от зависания: у CliffWalking нет лимита шагов по
    # умолчанию (в отличие от FrozenLake), а падение в обрыв не завершает эпизод,
    # только возвращает агента в старт
    env = make_env(render_mode="human")
    state, info = env.reset()
    done = False
    terminated = False
    total_reward = 0
    steps = 0

    # временно убираем исследование, хотим увидеть выученную политику,
    # а не случайные метания
    old_epsilon = agent.epsilon
    agent.epsilon = 0.0

    while not done and steps < max_steps:
        action = agent.choose_action(state)
        state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward
        steps += 1
        time.sleep(delay)

    agent.epsilon = old_epsilon  # возвращаем как было, вдруг агент еще понадобится
    env.close()

    # в CliffWalking награда за шаг к цели тоже -1, поэтому успех определяем
    # по terminated (дошел до цели), а не по знаку total_reward, как во FrozenLake
    result = "успех, дошёл до цели!" if terminated else "не дошёл за отведённое число шагов"
    print(f"Эпизод завершён за {steps} шагов, награда {total_reward:.0f}, {result}")
