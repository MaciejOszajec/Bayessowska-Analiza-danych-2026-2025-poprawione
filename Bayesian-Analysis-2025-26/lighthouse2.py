import numpy as np
import matplotlib.pyplot as plt
from scipy.special import logsumexp

flash_x = np.loadtxt("lighthouse_2d.txt")

x_min, x_max = -130.0, -90.0
h_min, h_max = 1.0, 10.0
nx, nh = 450, 300

x_lhs = np.linspace(x_min, x_max, nx)
hs = np.linspace(h_min, h_max, nh)
x_grid, h_grid = np.meshgrid(x_lhs, hs)


def p(x_lh, h, x):
    return (h / np.pi) / ((x - x_lh) ** 2 + h ** 2)


def log_p(x_lh, h, x):
    return np.log(h / np.pi) - np.log((x - x_lh) ** 2 + h ** 2)


def normalize(values):
    return values / np.sum(values)


def log_p_many(x_lh, h_lh, flashes):
    log_like = np.zeros_like(x_lh, dtype=float)
    for flash in np.atleast_1d(flashes):
        log_like += log_p(x_lh, h_lh, flash)
    return log_like


def posterior_log(flashes):
    log_post = log_p_many(x_grid, h_grid, flashes)
    log_post -= logsumexp(log_post)
    return np.exp(log_post)


def posterior_one_flash(x):
    return normalize(p(x_grid, h_grid, x))


def map_from_posterior(posterior):
    i, j = np.unravel_index(np.argmax(posterior), posterior.shape)
    return float(x_lhs[j]), float(hs[i])


def marginals(posterior):
    marginal_x = posterior.sum(axis=0)
    marginal_h = posterior.sum(axis=1)
    return normalize(marginal_x), normalize(marginal_h)


def marginal_maps(posterior):
    marginal_x, marginal_h = marginals(posterior)
    return float(x_lhs[np.argmax(marginal_x)]), float(hs[np.argmax(marginal_h)])


def plot_posterior(posterior, title):
    mx, mh = map_from_posterior(posterior)
    plt.figure(figsize=(9, 6))
    contour = plt.contourf(x_lhs, hs, posterior, levels=35)
    plt.contour(x_lhs, hs, posterior, colors="black", linewidths=0.4, levels=20)
    plt.scatter([mx], [mh], marker="x", s=90, label=f"MAP = ({mx:.2f}, {mh:.2f})")
    plt.xlabel("x_lh")
    plt.ylabel("h")
    plt.title(title)
    plt.colorbar(contour, label="posterior")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_marginals(posterior, title):
    marginal_x, marginal_h = marginals(posterior)
    map_x, map_h = marginal_maps(posterior)

    plt.figure(figsize=(9, 5))
    plt.plot(x_lhs, marginal_x)
    plt.axvline(map_x, linestyle="--", label=f"MAP = {map_x:.2f}")
    plt.xlabel("x_lh")
    plt.ylabel("posterior probability")
    plt.title(title + ": marginal over x_lh")
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(9, 5))
    plt.plot(hs, marginal_h)
    plt.axvline(map_h, linestyle="--", label=f"MAP = {map_h:.2f}")
    plt.xlabel("h")
    plt.ylabel("posterior probability")
    plt.title(title + ": marginal over h")
    plt.legend()
    plt.tight_layout()
    plt.show()


def print_summary(title, posterior):
    joint_map = map_from_posterior(posterior)
    marginal_map = marginal_maps(posterior)
    print(title)
    print(f"  joint MAP:      x_lh = {joint_map[0]:.4f}, h = {joint_map[1]:.4f}")
    print(f"  marginal MAPs:  x_lh = {marginal_map[0]:.4f}, h = {marginal_map[1]:.4f}\n")


def main():
    posterior1 = posterior_one_flash(flash_x[0])
    posterior1_log = posterior_log(flash_x[0])
    print("Problem 1: p, log_p and log_p_many implemented")
    print(f"Check plain/log after one flash: {np.max(np.abs(posterior1 - posterior1_log)):.3e}\n")
    print_summary("Problem 2: one flash", posterior1)
    plot_posterior(posterior1, "Posterior after one flash")
    plot_marginals(posterior1, "One flash")

    posterior2 = posterior_log(flash_x[:2])
    print_summary("Problem 3.2: two flashes", posterior2)
    plot_posterior(posterior2, "Posterior after two flashes")
    plot_marginals(posterior2, "Two flashes")

    posterior_all = posterior_log(flash_x)
    print_summary("Problem 3.3: all flashes", posterior_all)
    plot_posterior(posterior_all, "Posterior for all flashes")
    plot_marginals(posterior_all, "All flashes")


if __name__ == "__main__":
    main()
