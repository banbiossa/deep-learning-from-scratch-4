import abc
import logging
from collections import defaultdict

import numpy as np
import typer
from deep_learning_from_scratch_4.ch04.gridworld import GridWorld
from tqdm import tqdm


class BaseAgent:
    def __init__(self) -> None:
        self.gamma = 0.9
        self.action_size = 4

        random_actions = {0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25}
        self.pi = defaultdict(lambda: random_actions)
        self.Q = defaultdict(lambda: 0)
        self.cnts = defaultdict(lambda: 0)
        self.memory = []

    def get_action(self, state):
        action_probs = self.pi[state]
        actions = list(action_probs.keys())
        probs = list(action_probs.values())
        return np.random.choice(actions, p=probs)

    def add(self, state, action, reward):
        data = (state, action, reward)
        self.memory.append(data)

    def reset(self):
        self.memory.clear()

    @abc.abstractmethod
    def update(self):
        pass


class RandomAgent(BaseAgent):
    def update(self):
        G = 0
        for data in reversed(self.memory):
            state, action, reward = data
            G = self.gamma * G + reward
            self.cnts[state] += 1
            self.Q[state] += (G - self.Q[state]) / self.cnts[state]


def get_agent(agent_name: str) -> BaseAgent:
    if agent_name == "random":
        return RandomAgent()
    if agent_name == "mc":
        return McAgent()

    # return random agent as base case default
    logging.warning(f"Unknown agent name: {agent_name}. Using random agent")
    return RandomAgent()


def greedy_probs(Q, state, action_size=1):
    qs = [Q[(state, action)] for action in range(action_size)]
    max_action = np.argmax(qs)

    action_probs = {action: 0.0 for action in range(action_size)}
    action_probs[max_action] = 1.0
    return action_probs


def epsilon_greedy(Q, state, epsilon=0.0, action_size=4):
    qs = [Q[(state, action)] for action in range(action_size)]
    max_action = np.argmax(qs)

    base_prob = epsilon / action_size
    action_probs = {action: base_prob for action in range(action_size)}
    action_probs[max_action] += 1 - epsilon
    return action_probs


class McAgent(BaseAgent):
    def __init__(self) -> None:
        super().__init__()
        self.alpha = 0.1
        self.epsilon = 0.1

    def update(self):
        G = 0
        for data in reversed(self.memory):
            state, action, reward = data
            G = self.gamma * G + reward
            key = (state, action)
            self.Q[key] += (G - self.Q[key]) * self.alpha

            self.pi[state] = epsilon_greedy(self.Q, state, self.epsilon)


def play(agent_name="random"):
    env = GridWorld()
    agent = get_agent(agent_name)

    episodes = 1_000
    for episode in tqdm(range(episodes)):
        state = env.reset()
        agent.reset()

        while True:
            action = agent.get_action(state)
            next_state, reward, done = env.step(action)

            agent.add(state, action, reward)
            if done:
                agent.update()
                break
            state = next_state

    if agent_name == "random":
        env.render_v(agent.Q)
    else:
        env.render_q(agent.Q)


if __name__ == "__main__":
    typer.run(play)
