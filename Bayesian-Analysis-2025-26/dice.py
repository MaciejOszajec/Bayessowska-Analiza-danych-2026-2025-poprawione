import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.optimize import fsolve

# Counts of faces 1, 2, 3, 4, 5, 6.
data = np.asarray([181, 164, 154, 174, 172, 155], dtype=int)


def beta_from_mean_std(mean, std):
    def equations(par):
        a, b = par
        return [
            a / (a + b) - mean,
            a * b / ((a + b) ** 2 * (a + b + 1)) - std ** 2,
        ]

    return fsolve(equations, [10.0, 50.0])


def hdr_grid(distribution, xmin=0.0, xmax=0.5, prob=0.90, n_grid=8000):
    x = np.linspace(xmin, xmax, n_grid)
    pdf = distribution.pdf(x)
    dx = x[1] - x[0]

    order = np.argsort(pdf)[::-1]
    mass = np.cumsum(pdf[order] * dx)
    last = np.searchsorted(mass, prob)
    selected = np.sort(order[: last + 1])
    return float(x[selected[0]]), float(x[selected[-1]])


def posterior_summary(alpha0, beta0, title):
    successes = int(data[5])
    failures = int(data[:5].sum())

    alpha = alpha0 + successes
    beta = beta0 + failures
    posterior = stats.beta(alpha, beta)

    if alpha > 1 and beta > 1:
        mode = (alpha - 1.0) / (alpha + beta - 2.0)
    else:
        mode = np.nan

    mean = posterior.mean()
    median = posterior.isf(0.5)
    hdr = hdr_grid(posterior, 0.0, 0.5, 0.90)

    print(title)
    print(f"  alpha = {alpha:.6g}, beta = {beta:.6g}")
    print(f"  MAP    = {mode:.6f}")
    print(f"  mean   = {mean:.6f}")
    print(f"  median = {median:.6f}")
    print(f"  90% HDR = [{hdr[0]:.6f}, {hdr[1]:.6f}]\n")

    x = np.linspace(0.0, 0.5, 1000)
    plt.figure(figsize=(8, 5))
    plt.plot(x, posterior.pdf(x), label="posterior")
    plt.axvline(mode, linestyle="--", label=f"MAP = {mode:.4f}")
    plt.axvline(mean, linestyle=":", label=f"mean = {mean:.4f}")
    plt.axvline(median, linestyle="-.", label=f"median = {median:.4f}")
    plt.axvspan(hdr[0], hdr[1], alpha=0.20, label="90% HDR")
    plt.xlabel("p_6")
    plt.ylabel("density")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()

    return posterior, hdr


def find_number_of_throws(alpha0, beta0, epsilon=0.01, prob=0.90):
    p0 = 1.0 / 6.0
    p_true = p0 + epsilon

    for n in range(101, 200000):
        k = int(round(p_true * n))
        posterior = stats.beta(alpha0 + k, beta0 + n - k)
        hdr = hdr_grid(posterior, 0.05, 0.30, prob)
        if p0 < hdr[0] or p0 > hdr[1]:
            return n, k, hdr

    return None, None, None


def main():
    if data.shape != (6,):
        raise ValueError("data must contain six counts")
    if data.sum() <= 100:
        print("Warning: the task asks for more than 100 throws.")

    faces = np.arange(1, 7)
    plt.figure(figsize=(6, 4))
    plt.bar(faces, data)
    plt.xlabel("face")
    plt.ylabel("count")
    plt.title("Dice results")
    plt.tight_layout()
    plt.show()

    posterior_summary(1.0, 1.0, "Problem 1: uniform prior")

    alpha0, beta0 = beta_from_mean_std(1.0 / 6.0, 1.0 / 100.0)
    print(f"Beta prior parameters: alpha = {alpha0:.6f}, beta = {beta0:.6f}\n")
    posterior_summary(alpha0, beta0, "Problem 2: beta prior")

    n_uniform, k_uniform, hdr_uniform = find_number_of_throws(1.0, 1.0)
    n_beta, k_beta, hdr_beta = find_number_of_throws(alpha0, beta0)

    print("Problem 3: crooked dice, epsilon = 0.01")
    print(f"  uniform prior: n = {n_uniform}, expected sixes = {k_uniform}, 90% HDR = {hdr_uniform}")
    print(f"  beta prior:    n = {n_beta}, expected sixes = {k_beta}, 90% HDR = {hdr_beta}")


if __name__ == "__main__":
    main()
