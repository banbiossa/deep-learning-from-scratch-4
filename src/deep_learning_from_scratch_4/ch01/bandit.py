import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


class Bandit:
    def __init__(self, arms=10):
        self.rates = np.random.rand(arms)

    def reset(self):
        self.rates = np.random.rand(len(self.rates))

    def play(self, arm):
        rate = self.rates[arm]
        if rate > np.random.rand():
            return 1
        return 0


class Agent:
    def __init__(self, epsilon, action_size=10):
        self.episilon = epsilon
        self.action_size = action_size
        self.reset()
        # self.Qs = np.zeros(action_size)
        # self.ns = np.zeros(action_size)

    def reset(self):
        self.Qs = np.zeros(self.action_size)
        self.ns = np.zeros(self.action_size)

    def update(self, action, reward):
        self.ns[action] += 1
        self.Qs[action] += (reward - self.Qs[action]) / self.ns[action]

    def get_action(self):
        if np.random.rand() < self.episilon:
            return np.random.randint(0, len(self.Qs))
        return np.argmax(self.Qs)


def play2():
    steps = 1000
    epsilon = 0.1
    bandit = Bandit()
    agent = Agent(epsilon)
    total_reward = 0
    total_rewards = []
    rates = []

    for step in tqdm(range(steps)):
        action = agent.get_action()
        reward = bandit.play(action)
        agent.update(action, reward)
        total_reward += reward

        total_rewards.append(total_reward)
        rates.append(total_reward / (step + 1))

    print(total_reward)

    plt.ylabel("total reward")
    plt.xlabel("steps")
    plt.plot(total_rewards)
    plt.show()

    plt.ylabel("rates")
    plt.xlabel("steps")
    plt.plot(rates)
    plt.show()

    print(bandit.rates)


def play():
    np.random.seed(0)
    bandit = Bandit()
    for i in range(3):
        print(bandit.play(0))


if __name__ == "__main__":
    play()
    play2()
