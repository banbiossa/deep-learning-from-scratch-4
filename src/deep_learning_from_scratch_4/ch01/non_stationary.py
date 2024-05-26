import matplotlib.pyplot as plt
import numpy as np

from deep_learning_from_scratch_4.ch01.bandit import Agent
from deep_learning_from_scratch_4.ch01.bandit_avg import play


class NonStationaryBandit:
    def __init__(self, arms=10):
        self.arms = arms
        self.reset()
        # self.rates = np.random.rand(arms)

    def reset(self):
        self.rates = np.random.rand(self.arms)

    def play(self, arm) -> int:
        rate = self.rates[arm]
        self.rates += 0.1 * np.random.randn(self.arms)
        if rate > np.random.rand():
            return 1
        return 0


class AlphaAgent:
    def __init__(self, epsilon, alpha, actions=10):
        self.epsilon = epsilon
        self.actions = actions
        self.alpha = alpha
        self.reset()  # self.Qs = np.zeros(actions)

    def reset(self):
        self.Qs = np.zeros(self.actions)

    def update(self, action, reward):
        self.Qs[action] += (reward - self.Qs[action]) * self.alpha

    def get_action(self):
        if np.random.rand() < self.epsilon:
            return np.random.randint(0, len(self.Qs))
        return np.argmax(self.Qs)


def play_here():
    alpha = play(
        agent=AlphaAgent(epsilon=0.1, alpha=0.9),
        bandit=NonStationaryBandit(),
        runs=200,
        steps=1000,
    )
    constant = play(
        agent=Agent(epsilon=0.1),
        bandit=NonStationaryBandit(),
        runs=200,
        steps=1000,
    )

    plt.ylabel("average reward")
    plt.xlabel("steps")
    plt.plot(alpha)
    plt.plot(constant)
    plt.legend(["alpha=0.9", "constant"])
    plt.show()


if __name__ == "__main__":
    play_here()
