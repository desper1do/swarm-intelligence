import numpy as np
import matplotlib.pyplot as plt


def moving_average(data, w):
    return np.convolve(data, np.ones(w) / w, mode='valid')


def plot_comparison(results, window=100):
    plt.figure(figsize=(10, 6))
    plt.plot(moving_average(results["sarsa"]["rewards"], window), label='SARSA')
    plt.plot(moving_average(results["qlearning"]["rewards"], window), label='Q-Learning')
    plt.plot(moving_average(results["reinforce"]["rewards"], window), label='REINFORCE')
    plt.xlabel("Эпизод")
    plt.ylabel(f"Награда (скользящее среднее, окно={window})")
    plt.title("Сравнение алгоритмов на Taxi-v4")
    plt.legend()
    plt.show()