import numpy as np
import pymc as pm
import arviz as az

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

flash_x = np.loadtxt("lighthouse.txt")
flash_x_2d = np.loadtxt("lighthouse_2d.txt")


def hdi_pair(samples, prob=0.95):
    interval = az.hdi(samples, hdi_prob=prob)
    return float(interval[0]), float(interval[1])


def problem_1():
    h = 1.0

    with pm.Model() as model:
        x0 = pm.Normal("x0", mu=0.0, sigma=20.0)
        pm.Cauchy("obs", alpha=x0, beta=h, observed=flash_x)

        map_est = pm.find_MAP(progressbar=False)
        idata = pm.sample(
            draws=1500,
            tune=1000,
            chains=2,
            cores=1,
            random_seed=RANDOM_SEED,
            target_accept=0.90,
            progressbar=True,
        )

    samples = az.extract(idata, var_names=["x0"])["x0"].values
    mean = float(samples.mean())
    low, high = hdi_pair(samples, 0.95)

    print("Problem 1: lighthouse position, h = 1")
    print(f"  MAP x0 = {float(map_est['x0']):.2f}")
    print(f"  posterior mean x0 = {mean:.2f}")
    print(f"  95% HDI for x0 = [{low:.2f}, {high:.2f}]\n")


def problem_2():
    with pm.Model() as model:
        x0 = pm.Normal("x0", mu=-110.0, sigma=30.0)
        h = pm.HalfNormal("h", sigma=20.0)
        pm.Cauchy("obs", alpha=x0, beta=h, observed=flash_x_2d)

        map_est = pm.find_MAP(progressbar=False)
        idata = pm.sample(
            draws=1500,
            tune=1000,
            chains=2,
            cores=1,
            random_seed=RANDOM_SEED,
            target_accept=0.90,
            progressbar=True,
        )

    samples = az.extract(idata, var_names=["x0", "h"])
    x0_samples = samples["x0"].values
    h_samples = samples["h"].values

    x_low, x_high = hdi_pair(x0_samples, 0.95)
    h_low, h_high = hdi_pair(h_samples, 0.95)

    print("Problem 2: lighthouse position and distance")
    print(f"  MAP x0 = {float(map_est['x0']):.2f}")
    print(f"  MAP h  = {float(map_est['h']):.2f}")
    print(f"  posterior mean x0 = {float(x0_samples.mean()):.2f}")
    print(f"  95% HDI for x0 = [{x_low:.2f}, {x_high:.2f}]")
    print(f"  posterior mean h  = {float(h_samples.mean()):.2f}")
    print(f"  95% HDI for h  = [{h_low:.2f}, {h_high:.2f}]")


def main():
    problem_1()
    problem_2()


if __name__ == "__main__":
    main()
