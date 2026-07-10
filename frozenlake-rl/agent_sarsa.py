import numpy as np
from config import ALPHA, GAMMA, EPSILON_START, EPSILON_MIN, EPSILON_DECAY

class SARSAAgent:
    def __init__(self, n_states, n_actions):
        # Шаг 1: инициализация Q-таблицы нулями, размер |S| x |A| (как и в Q-Learning)
        self.q_table = np.zeros((n_states, n_actions))
        self.n_actions = n_actions
        self.alpha = ALPHA
        self.gamma = GAMMA
        self.epsilon = EPSILON_START

    def choose_action(self, state):
        # epsilon-greedy выбор действия - тот же принцип, что и в Q-Learning
        t = np.random.rand()
        if t < self.epsilon:
            return np.random.randint(self.n_actions)
        else:
            return np.argmax(self.q_table[state])

    def update(self, state, action, reward, next_state, next_action, done):
        # Шаг 5: обновление Q(s,a) по формуле SARSA
        # on-policy: цель строится по Q(s',a') - действию a', которое агент
        # реально выберет по своей epsilon-greedy политике (в т.ч. может быть случайным),
        # а не по максимуму как в Q-Learning - отсюда более осторожное поведение
        if done:
            td_target = reward
        else:
            td_target = reward + self.gamma * self.q_table[next_state][next_action]
        td_error = td_target - self.q_table[state][action]
        self.q_table[state][action] += self.alpha * td_error

    def decay_epsilon(self):
        # Шаг 7: уменьшение epsilon после эпизода
        self.epsilon = max(EPSILON_MIN, self.epsilon * EPSILON_DECAY)