from collections import defaultdict

from deep_learning_from_scratch_4.ch04.gridworld import GridWorld
from deep_learning_from_scratch_4.ch04.policy_eval import policy_eval


def argmax(d: dict):
    max_value = max(d.values())
    max_key = 0
    for key, value in d.items():
        if value == max_value:
            max_key = key
    return max_key


def greedy_policy(V: defaultdict, env: GridWorld, gamma):
    pi = {}

    for state in env.states():
        action_values = {}

        for action in env.actions():
            next_state = env.next_state(state, action)
            r = env.reward(state, action, next_state)
            value = r + gamma * V[next_state]
            action_values[action] = value

        max_action = argmax(action_values)
        action_probs = {0: 0, 1: 0, 2: 0, 3: 0}
        action_probs[max_action] = 1
        pi[state] = action_probs
    return pi


def policy_iter(env: GridWorld, gamma, threshold=1e-3, is_render=False):
    pi = defaultdict(lambda: {0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25})
    V = defaultdict(lambda: 0)

    while True:
        V = policy_eval(pi, V, env, threshold)
        new_pi = greedy_policy(V, env, gamma)

        if is_render:
            env.render_v(V, new_pi)

        if new_pi == pi:
            break
        pi = new_pi
    return pi


def play():
    env = GridWorld()
    gamma = 0.9
    pi = policy_iter(env, gamma, is_render=True)


if __name__ == "__main__":
    play()
