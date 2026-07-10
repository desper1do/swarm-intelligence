import numpy as np
from config import ALPHA, GAMMA, EPSILON_START, EPSILON_MIN, EPSILON_DECAY

class QLearningAgent:
    def __init__(self, n_states, n_actions):
        # Шаг 1: инициализация Q-таблицы нулями, размер |S| x |A|
        self.q_table = np.zeros((n_states, n_actions))
        self.n_actions = n_actions
        self.gamma = GAMMA
        self.alpha = ALPHA
        self.epsilon = EPSILON_START


    def choose_action(self, state):
        # Шаг 3: epsilon-greedy - с вероятностью epsilon случайное действие (исследование),
        # иначе жадное по Q-таблице (использование)
        t = np.random.rand()
        if t < self.epsilon:
            return np.random.randint(self.n_actions)
        else:
            return np.argmax(self.q_table[state])


    def update(self, state, action, reward, next_state, done):
        # Шаг 5: обновление Q(s,a) по формуле Q-Learning
        # off-policy: цель строится по max Q(s',a') - лучшему действию в s',
        # а не по тому, что агент реально сделает дальше (в этом отличие от SARSA)
        if done:
            td_target = reward
        else:
            best_next_action = np.argmax(self.q_table[next_state])
            td_target = reward + self.gamma * self.q_table[next_state][best_next_action]
        td_error = td_target - self.q_table[state][action]
        self.q_table[state][action] += self.alpha * td_error


    def decay_epsilon(self):
        # Шаг 7: уменьшение epsilon - агент постепенно переходит от исследования к использованию
        self.epsilon = max(EPSILON_MIN, self.epsilon * EPSILON_DECAY)
