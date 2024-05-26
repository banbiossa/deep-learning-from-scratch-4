import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from deep_learning_from_scratch_4.ch01.bandit import Agent, Bandit


def play(bandit=Bandit(), agent=Agent(epsilon=0.1), runs=200, steps=1000):
    # runs = 200
    # steps = 1000
    # epsilon = 0.1
    all_rates = np.zeros((runs, steps))

    for run in tqdm(range(runs)):
        # bandit = bandit_class()
        # agent = agent_class(epsilon)
        agent.reset()
        bandit.reset()
        total_reward = 0
        rates = []

        for step in range(steps):
            action = agent.get_action()
            reward = bandit.play(action)
            agent.update(action, reward)
            total_reward += reward
            rates.append(total_reward / (step + 1))

        all_rates[run] = rates

    avg_rates = np.average(all_rates, axis=0)
    plt.ylabel("average reward")
    plt.xlabel("steps")
    plt.plot(avg_rates)
    plt.show()

    return avg_rates


if __name__ == "__main__":
    play()
