from collections import defaultdict

import numpy as np
from deep_learning_from_scratch_4.ch05.mc_eval import epsilon_greedy


class QLearningAgent:
    def __init__(self):
        self.gamma = 0.9
        self.alpha = 0.8
        self.epsilon = 0.1
        self.action_size = 4

        random_actions = {0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25}
        self.b = defaultdict(lambda: random_actions)  # behavior policy
        self.Q = defaultdict(lambda: 0)

    def get_action(self, state):
        self.b[state] = epsilon_greedy(self.Q, state, self.epsilon)
        action_probs = self.b[state]
        actions = list(action_probs.keys())
        probs = list(action_probs.values())
        return np.random.choice(actions, p=probs)

    def update(self, state, action, reward, next_state, done):
        if done:
            next_q_max = 0
        else:
            next_qs = [self.Q[next_state, a] for a in range(self.action_size)]
            next_q_max = max(next_qs)

        target = self.gamma * next_q_max + reward
        self.Q[state, action] += self.alpha * (target - self.Q[state, action])
