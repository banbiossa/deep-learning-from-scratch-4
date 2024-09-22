import copy

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from deep_learning_from_scratch_4.ch08.replay_buffer import ReplayBuffer


class QNet(nn.Module):
    def __init__(self, action_size):
        super().__init__()
        self.l1 = nn.Linear(4, 128)
        self.l2 = nn.Linear(128, 128)
        self.l3 = nn.Linear(128, action_size)

    def forward(self, x):
        x = nn.ReLU()(self.l1(x))
        x = nn.ReLU()(self.l2(x))
        x = self.l3(x)
        return x


class DQNAgent:
    def __init__(self) -> None:
        self.gamma = 0.9
        self.lr = 0.0005
        self.epsilon = 0.1
        self.buffer_size = 10_000
        self.batch_size = 32
        self.action_size = 2

        self.replay_buffer = ReplayBuffer(self.buffer_size, self.batch_size)
        self.qnet = QNet(self.action_size)
        self.qnet_target = QNet(self.action_size)
        self.optimizer = torch.optim.Adam(self.qnet.parameters(), lr=self.lr)

    def sync_qnet(self):
        self.qnet_target = copy.deepcopy(self.qnet)

    def get_action(self, state) -> int:
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.action_size)
        else:
            state = state[np.newaxis, :]
            state = torch.tensor(state, dtype=torch.float32)
            qs = self.qnet(state)
            return qs.data.argmax().item()

    def update(self, state, action, reward, next_state, done):
        self.replay_buffer.add(state, action, reward, next_state, done)
        if len(self.replay_buffer) < self.batch_size:
            return

        state, action, reward, next_state, done = self.replay_buffer.get_batch()
        state = torch.tensor(state, dtype=torch.float32)
        qs = self.qnet(state)
        q = qs[np.arange(self.batch_size), action]

        next_state = torch.tensor(next_state, dtype=torch.float32)
        next_qs = self.qnet_target(next_state)
        next_q = next_qs.max(dim=1)[0].detach()
        reward = torch.tensor(reward, dtype=torch.float32)
        done = torch.tensor(done, dtype=torch.float)
        target = reward + (1 - done) * self.gamma * next_q

        loss = nn.MSELoss()(q, target)

        self.qnet.zero_grad()
        loss.backward()
        self.optimizer.step()


def main():
    episodes = 300
    sync_interval = 20
    env = gym.make("CartPole-v1")
    agent = DQNAgent()
    reward_history = []

    for episode in tqdm(range(episodes)):
        state, _ = env.reset()
        done = False
        total_reward = 0

        while not done:
            action = agent.get_action(state)
            next_state, reward, done, _, info = env.step(action)

            agent.update(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

        if episode % sync_interval == 0:
            agent.sync_qnet()

        reward_history.append(total_reward)

    plt.plot(reward_history)
    plt.show()

    # move greedy
    agent.epsilon = 0
    env = gym.make("CartPole-v1", render_mode="human")
    state, _ = env.reset()
    done = False
    total_reward = 0

    while not done:
        action = agent.get_action(state)
        next_state, reward, done, _, info = env.step(action)
        state = next_state
        total_reward += reward
        env.render()

    print(f"total {total_reward}")


if __name__ == "__main__":
    main()
