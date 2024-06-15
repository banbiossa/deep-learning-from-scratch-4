from deep_learning_from_scratch_4.ch04.gridworld import GridWorld
from deep_learning_from_scratch_4.ch05.mc_eval import BaseAgent
from tqdm import tqdm


class TdAgent(BaseAgent):
    def __init__(self) -> None:
        super().__init__()
        self.alpha = 0.01

    def eval(self, state, reward, next_state, done):
        next_Q = 0 if done else self.Q[next_state]
        target = reward + self.gamma * next_Q

        self.Q[state] += (target - self.Q[state]) * self.alpha


def play():
    env = GridWorld()
    agent = TdAgent()

    episodes = 1_000
    for episode in tqdm(range(episodes)):
        state = env.reset()
        # agent.reset()

        while True:
            action = agent.get_action(state)
            next_state, reward, done = env.step(action)

            agent.eval(state, reward, next_state, done)
            if done:
                break
            state = next_state

    env.render_v(agent.Q)


if __name__ == "__main__":
    play()
