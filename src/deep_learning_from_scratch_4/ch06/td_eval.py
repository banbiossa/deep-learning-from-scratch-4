from deep_learning_from_scratch_4.ch05.mc_eval import BaseAgent


class TdAgent(BaseAgent):
    def __init__(self) -> None:
        super().__init__()
        self.alpha = 0.01

    def eval(self, state, reward, next_state, done):
        next_Q = 0 if done else self.Q[next_state]
        target = reward + self.gamma * next_Q

        self.Q[state] += (target - self.Q[state]) * self.alpha
