import numpy as np


def play():
    x = np.array([1, 2, 3])
    pi = np.array([0.1, 0.2, 0.7])

    e = np.sum(x * pi)
    print(f"Expected value: {e:.2f}")

    # mc
    n = 100
    samples = []
    for _ in range(n):
        s = np.random.choice(x, p=pi)
        samples.append(s)

    mean = np.mean(samples)
    var = np.var(samples)
    print(f"MC mean: {mean:.2f}, var: {var:.2f}")


def do(b):
    x = np.array([1, 2, 3])
    pi = np.array([0.1, 0.2, 0.7])
    n = 100
    samples = []

    for _ in range(n):
        idx = np.arange(len(b))
        i = np.random.choice(idx, p=b)
        s = x[i]
        rho = pi[i] / b[i]
        samples.append(s * rho)

    mean = np.mean(samples)
    var = np.var(samples)
    print(f"IS mean: {mean:.2f}, var: {var:.2f}")


if __name__ == "__main__":
    play()
    do(b=np.array([1 / 3, 1 / 3, 1 / 3]))
    do(b=np.array([0.2, 0.2, 0.6]))
