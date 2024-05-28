import numpy as np
from deep_learning_from_scratch_4.ch04.gridworld import GridWorld


def play():
    env = GridWorld()
    env.render_v()
    V = {}
    for state in env.states():
        V[state] = np.random.randn()
    env.render_v(V)


if __name__ == "__main__":
    play()
