import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import pymc as pm
import arviz as az

SEED = 1412345
np.random.seed(SEED)


def load_regression_data(path="regression.txt"):
    data = np.loadtxt(path)
    if data.ndim == 1:
        data = data.reshape((-1, 3))
    if data.shape[1] < 2:
        raise ValueError("regression data must contain at least two columns")
    return data[:, 0], data[:, 1]


def quadratic(x, a, b, c):
    return a * x ** 2 + b * x + c


def main():
    x, y = load_regression_data("regression.txt")

    popt, pcov = curve_fit(quadratic, x, y)
    a_cf, b_cf, c_cf = popt
    mse_cf = float(np.mean((y - quadratic(x, *popt)) ** 2))

    print("Problem 1: scipy curve_fit")
    print(f"  a = {a_cf:.6f}, b = {b_cf:.6f}, c = {c_cf:.6f}")
    print(f"  MSE = {mse_cf:.6f}\n")

    with pm.Model() as model:
        a = pm.Normal("a", mu=0.0, sigma=10.0)
        b = pm.Normal("b", mu=0.0, sigma=10.0)
        c = pm.Normal("c", mu=0.0, sigma=10.0)
        sigma = pm.HalfFlat("sigma")
        pm.Potential("sigma_prior", -pm.math.log(sigma))

        mu = a * x ** 2 + b * x + c
        pm.Normal("y_obs", mu=mu, sigma=sigma, observed=y)

        start = {
            "a": float(a_cf),
            "b": float(b_cf),
            "c": float(c_cf),
            "sigma": float(np.std(y - quadratic(x, *popt), ddof=1)),
        }
        map_est = pm.find_MAP(start=start, progressbar=False)
        idata = pm.sample(
            draws=2000,
            tune=1000,
            chains=2,
            cores=1,
            target_accept=0.90,
            random_seed=SEED,
            progressbar=True,
        )

    print("Problem 2: PyMC MAP")
    print(
        f"  a = {map_est['a']:.6f}, b = {map_est['b']:.6f}, "
        f"c = {map_est['c']:.6f}, sigma = {map_est['sigma']:.6f}\n"
    )

    posterior = idata.posterior
    a_s = posterior["a"].values.reshape(-1)
    b_s = posterior["b"].values.reshape(-1)
    c_s = posterior["c"].values.reshape(-1)
    sigma_s = posterior["sigma"].values.reshape(-1)

    samples = np.vstack([a_s, b_s, c_s, sigma_s])
    corr = np.corrcoef(samples)
    print("Correlation matrix for a, b, c, sigma")
    print(np.array2string(corr, precision=3, suppress_small=True))
    print()

    x_test = np.linspace(0.0, 2.0, 100)
    values = (
        a_s[:, None] * x_test[None, :] ** 2
        + b_s[:, None] * x_test[None, :]
        + c_s[:, None]
    )
    mean_prediction = values.mean(axis=0)
    std_prediction = values.std(axis=0)

    plt.figure(figsize=(8, 5))
    plt.scatter(x, y, s=18, label="data")
    plt.plot(x_test, quadratic(x_test, *popt), label="curve_fit")
    plt.plot(x_test, mean_prediction, label="posterior mean")
    plt.fill_between(
        x_test,
        mean_prediction - std_prediction,
        mean_prediction + std_prediction,
        alpha=0.30,
        label="posterior mean ± 1 sd",
    )
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Quadratic regression")
    plt.legend()
    plt.tight_layout()
    plt.show()

    print("Prediction error on x_test")
    print(f"  mean standard deviation = {float(std_prediction.mean()):.6f}")
    print(f"  max standard deviation  = {float(std_prediction.max()):.6f}")


if __name__ == "__main__":
    main()
