import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt
from tqdm import tqdm

from deep_learning_from_scratch_4.ch04.gridworld import GridWorld


class QNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(12, 100)
        self.l2 = nn.Linear(100, 4)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = self.l2(x)
        return x


def one_hot(state):
    height, width = 3, 4
    vec = np.zeros(height * width)
    y, x = state
    idx = width * y + x
    vec[idx] = 1
    vec_batch = vec[np.newaxis, :]
    return torch.tensor(vec_batch, dtype=torch.float32)


class QLearningAgent:
    def __init__(self):
        self.gamma = 0.9
        self.lr = 0.01
        self.epsilon = 0.1
        self.action_size = 4

        self.qnet = QNet()
        self.optimizer = torch.optim.Adam(self.qnet.parameters(), lr=self.lr)

    def get_action(self, state) -> int:
        """_summary_

        Args:
            state (_type_): one hot encoded state

        Returns:
            int: best action
        """
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.action_size)

        qs = self.qnet(state)
        return qs.argmax().item()

    def update(self, state, action, reward, next_state, done):
        if done:
            next_q = torch.tensor(np.zeros(1), dtype=torch.float32)  # [0.]
        else:
            next_qs = self.qnet(next_state)
            next_q = next_qs.max(axis=1).values.detach()
            # next_q = next_qs.max().detach()

        target = self.gamma * next_q + reward
        qs = self.qnet(state)
        q = qs[:, action]

        self.optimizer.zero_grad()

        try:
            loss = F.mse_loss(target, q)
        except Exception:
            breakpoint()

        loss.backward()
        self.optimizer.step()

        return loss.data


def make_q(qnet):
    # make a dict of (state, action) -> q value from qnet
    # states are a tuple of (0-3, 0-4)
    states = [(i, j) for i in range(3) for j in range(4)]
    q_values = {}

    # we need a product of states and actions
    for state in states:
        qs = qnet(one_hot(state))

        # convert the qs to the (state, action) -> q value
        for action, q in enumerate(qs[0]):
            q_values[state, action] = q.item()

    return q_values


def test_make_q():
    qnet = QNet()
    Q = make_q(qnet)
    assert len(Q) == 12 * 4


def main():
    env = GridWorld()
    agent = QLearningAgent()

    episodes = 1000
    loss_history = []

    for episode in tqdm(range(episodes)):
        state = env.reset()
        state = one_hot(state)
        total_loss, cnt = 0, 0
        done = False

        while not done:
            action = agent.get_action(state)
            next_state, reward, done = env.step(action)
            next_state = one_hot(next_state)

            loss = agent.update(state, action, reward, next_state, done)
            total_loss += loss
            cnt += 1
            state = next_state

        avg_loss = total_loss / cnt
        loss_history.append(avg_loss)

    fix, ax = plt.subplots()
    plt.plot(loss_history)
    plt.show()

    # Q = make_q(agent.qnet)
    # env.render_q(Q)

    # visualize
    Q = {}
    for state in env.states():
        for action in env.action_space:
            q = agent.qnet(one_hot(state))[:, action]
            Q[state, action] = float(q.data)
    env.render_q(Q)


if __name__ == "__main__":
    test_make_q()
    main()
