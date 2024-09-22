import gymnasium as gym
import numpy as np


def main():
    env = gym.make("CartPole-v1", render_mode="human")
    state = env.reset()
    done = False

    while not done:
        env.render()
        action = np.random.choice([0, 1])
        next_state, reward, done, _, _ = env.step(action)
    env.close()


if __name__ == "__main__":
    main()
