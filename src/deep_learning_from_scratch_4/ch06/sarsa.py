from collections import defaultdict, deque

import numpy as np
from deep_learning_from_scratch_4.ch04.gridworld import GridWorld
from deep_learning_from_scratch_4.ch05.mc_eval import epsilon_greedy
from tqdm import tqdm


class SarsaAgent:
    def __init__(self) -> None:
        self.gamma = 0.9
        self.alpha = 0.8
        self.epsilon = 0.1
        self.action_size = 4

        random_actions = {0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25}
        self.pi = defaultdict(lambda: random_actions)
        self.Q = defaultdict(lambda: 0)
        self.memory = deque(maxlen=2)

    def get_actions(self, state):
        action_probs = self.pi[state]
        actions = list(action_probs.keys())
        probs = list(action_probs.values())
        return np.random.choice(actions, p=probs)

    def reset(self):
        self.memory.clear()

    def update(self, state, action, reward, done):
        self.memory.append((state, action, reward, done))
        if len(self.memory) < 2:
            return

        state, action, reward, done = self.memory[0]
        next_state, next_action, _, _ = self.memory[1]

        # next q
        next_q = 0 if done else self.Q[(next_state, next_action)]

        # td
        target = reward + self.gamma * next_q
        self.Q[state, action] += (target - self.Q[state, action]) * self.alpha

        # update policy
        self.pi[state] = epsilon_greedy(self.Q, state, self.epsilon)


def play():
    env = GridWorld()
    agent = SarsaAgent()

    episodes = 10_000
    for episode in tqdm(range(episodes)):
        state = env.reset()
        agent.reset()

        while True:
            action = agent.get_actions(state)
            next_state, reward, done = env.step(action)

            agent.update(state, action, reward, done)

            if done:
                agent.update(next_state, None, None, None)
                break
            state = next_state

    env.render_q(agent.Q)


if __name__ == "__main__":
    play()
