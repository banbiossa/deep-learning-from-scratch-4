from collections import defaultdict

from deep_learning_from_scratch_4.ch04.gridworld import GridWorld
from deep_learning_from_scratch_4.ch04.policy_iter import greedy_policy


def value_iter_onestep(V, env: GridWorld, gamma):
    for state in env.states():
        if state == env.goal_state:
            V[state] = 0
            continue

        action_values = []
        for action in env.actions():
            next_state = env.next_state(state, action)
            r = env.reward(state, action, next_state)
            value = r + gamma * V[next_state]
            action_values.append(value)

        V[state] = max(action_values)
    return V


def value_iter(V, env: GridWorld, gamma, threshold=1e-3, is_render=True):
    while True:
        if is_render:
            env.render_v(V)

        old_v = V.copy()
        V = value_iter_onestep(V, env, gamma)

        delta = 0
        for state in V.keys():
            t = abs(V[state] - old_v[state])
            if delta < t:
                delta = t

        if delta < threshold:
            break

    return V


def play():
    V = defaultdict(lambda: 0)
    env = GridWorld()
    gamma = 0.9

    V = value_iter(V, env, gamma)
    pi = greedy_policy(V, env, gamma)
    env.render_v(V, pi)


if __name__ == "__main__":
    play()
