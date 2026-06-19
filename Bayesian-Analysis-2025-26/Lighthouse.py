import numpy as np
import matplotlib.pyplot as plt
from scipy.special import logsumexp

h = 1.0
x_lim = 100.0
xs = np.linspace(-x_lim, x_lim, 5000)
flash_x = np.loadtxt("lighthouse.txt")


def cauchy_pdf(x, x_lh, h=1.0):
    return (h / np.pi) / ((x - x_lh) ** 2 + h ** 2)


def normalize(values):
    s = np.sum(values)
    if s == 0.0:
        raise ValueError("zero normalization constant")
    return values / s


def map_value(grid, posterior):
    return float(grid[np.argmax(posterior)])


def update_posterior(prior, measurement):
    return normalize(prior * cauchy_pdf(measurement, xs, h))


def posterior_many(grid, prior, measurements, h=1.0):
    log_post = np.log(prior)
    for measurement in measurements:
        log_post += np.log(h / np.pi) - np.log((measurement - grid) ** 2 + h ** 2)
    log_post -= logsumexp(log_post)
    return np.exp(log_post)


def symmetric_interval(grid, posterior, center, mass=0.95):
    distance = np.abs(grid - center)
    order = np.argsort(distance)
    cumulative = np.cumsum(posterior[order])
    radius = distance[order[np.searchsorted(cumulative, mass)]]
    return float(center - radius), float(center + radius)


def plot_posteriors(posteriors, labels, title):
    plt.figure(figsize=(10, 6))
    for posterior, label in zip(posteriors, labels):
        plt.plot(xs, posterior, label=str(label))
    plt.xlim(2, 8)
    plt.xlabel("x_lh")
    plt.ylabel("posterior probability")
    plt.title(title)
    plt.legend(title="measurement")
    plt.tight_layout()
    plt.show()


def main():
    prior = normalize(np.ones_like(xs))

    posterior1 = update_posterior(prior, flash_x[0])
    map1 = map_value(xs, posterior1)
    print(f"Problem 1: MAP after 1 measurement = {map1:.4f}")

    plt.figure(figsize=(10, 6))
    plt.plot(xs, posterior1)
    plt.xlim(2, 8)
    plt.xlabel("x_lh")
    plt.ylabel("posterior probability")
    plt.title("Posterior after one measurement")
    plt.tight_layout()
    plt.show()

    posterior2 = update_posterior(posterior1, flash_x[1])
    map2 = map_value(xs, posterior2)
    print(f"Problem 2: MAP after 2 measurements = {map2:.4f}")

    plt.figure(figsize=(10, 6))
    plt.plot(xs, posterior1, label="1 measurement")
    plt.plot(xs, posterior2, label="2 measurements")
    plt.xlim(2, 8)
    plt.xlabel("x_lh")
    plt.ylabel("posterior probability")
    plt.title("Posterior after two measurements")
    plt.legend()
    plt.tight_layout()
    plt.show()

    posterior = prior.copy()
    posteriors = []
    maps = []

    for measurement in flash_x[:100]:
        posterior = update_posterior(posterior, measurement)
        posteriors.append(posterior.copy())
        maps.append(map_value(xs, posterior))

    plot_posteriors(posteriors[:10], range(1, 11), "First 10 posterior distributions")
    plot_posteriors(posteriors[-10:], range(91, 101), "Last 10 posterior distributions")

    final_posterior = posteriors[-1]
    final_map = maps[-1]
    lower, upper = symmetric_interval(xs, final_posterior, final_map, 0.95)

    print(f"Problem 3: final MAP after 100 measurements = {final_map:.4f}")
    print(f"Problem 3: 95% symmetric interval around MAP = [{lower:.4f}, {upper:.4f}]")

    posterior_log = posterior_many(xs, prior, flash_x[:100], h)
    diff = np.max(np.abs(final_posterior - posterior_log))
    print(f"Problem 4: max difference iterative/log = {diff:.3e}")

    plt.figure(figsize=(10, 6))
    plt.plot(xs, final_posterior, label="iterative")
    plt.plot(xs, posterior_log, "--", label="log calculation")
    plt.xlim(2, 8)
    plt.xlabel("x_lh")
    plt.ylabel("posterior probability")
    plt.title("Final posterior")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
