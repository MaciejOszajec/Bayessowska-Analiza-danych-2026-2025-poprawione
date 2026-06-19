import numpy as np
import matplotlib.pyplot as plt
import pymc as pm
import arviz as az

DATA_PATH = "data.txt"
SEED = 42
DRAWS = 2000
TUNE = 1000
CHAINS = 2
CORES = 1
TARGET_ACCEPT = 0.90


def plot_discrete_hist(data, ax=None, **kwargs):
    if ax is None:
        ax = plt.gca()
    dmin = int(data.min())
    dmax = int(data.max())
    n_bins = dmax - dmin + 1
    return ax.hist(data, bins=n_bins, range=(dmin - 0.5, dmax + 0.5), **kwargs)


def fit_binomial(data, n):
    with pm.Model() as model:
        p = pm.Beta("p", alpha=1.0, beta=1.0)
        pm.Binomial("y", n=n, p=p, observed=data)
        idata = pm.sample(
            draws=DRAWS,
            tune=TUNE,
            chains=CHAINS,
            cores=CORES,
            target_accept=TARGET_ACCEPT,
            random_seed=SEED,
            progressbar=True,
        )
        ppc = pm.sample_posterior_predictive(idata, random_seed=SEED, progressbar=True)
        idata.extend(ppc)
    return idata


def fit_poisson(data):
    # Weakly informative prior centered near the sample mean.
    sample_mean = float(np.mean(data))
    alpha = 2.0
    beta = alpha / sample_mean

    with pm.Model() as model:
        lam = pm.Gamma("lam", alpha=alpha, beta=beta)
        pm.Poisson("y", mu=lam, observed=data)
        idata = pm.sample(
            draws=DRAWS,
            tune=TUNE,
            chains=CHAINS,
            cores=CORES,
            target_accept=TARGET_ACCEPT,
            random_seed=SEED,
            progressbar=True,
        )
        ppc = pm.sample_posterior_predictive(idata, random_seed=SEED, progressbar=True)
        idata.extend(ppc)
    return idata


def posterior_predictive_array(idata):
    return idata.posterior_predictive["y"].stack(sample=("chain", "draw")).values.T


def ppc_plot(data, predictive, title):
    fig, ax = plt.subplots(figsize=(8, 5))
    plot_discrete_hist(data, ax=ax, alpha=0.65, label="observed")
    plot_discrete_hist(predictive.ravel(), ax=ax, histtype="step", linewidth=1.5, label="posterior predictive")
    ax.set_xlabel("k")
    ax.set_ylabel("count")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    plt.show()


def ppc_summary(observed, simulated):
    return {
        "observed_mean": float(np.mean(observed)),
        "simulated_mean": float(np.mean(simulated)),
        "observed_variance": float(np.var(observed, ddof=1)),
        "simulated_variance": float(np.var(simulated, ddof=1)),
        "observed_min": int(np.min(observed)),
        "observed_max": int(np.max(observed)),
        "simulated_min": int(np.min(simulated)),
        "simulated_max": int(np.max(simulated)),
    }


def print_ppc_summary(name, summary):
    print(name)
    print(f"  mean: observed = {summary['observed_mean']:.2f}, simulated = {summary['simulated_mean']:.2f}")
    print(f"  variance: observed = {summary['observed_variance']:.2f}, simulated = {summary['simulated_variance']:.2f}")
    print(
        f"  range: observed = [{summary['observed_min']}, {summary['observed_max']}], "
        f"simulated = [{summary['simulated_min']}, {summary['simulated_max']}]\n"
    )


def main():
    data = np.loadtxt(DATA_PATH).astype(int)

    print(f"N = {len(data)}")
    print(f"mean = {np.mean(data):.3f}, sd = {np.std(data, ddof=1):.3f}, min = {data.min()}, max = {data.max()}\n")

    fig, ax = plt.subplots(figsize=(8, 5))
    plot_discrete_hist(data, ax=ax)
    ax.set_xlabel("k")
    ax.set_ylabel("count")
    ax.set_title("Observed data")
    fig.tight_layout()
    plt.show()

    idata35 = fit_binomial(data, n=35)
    idata45 = fit_binomial(data, n=45)
    idata_poisson = fit_poisson(data)

    ppc35 = posterior_predictive_array(idata35)
    ppc45 = posterior_predictive_array(idata45)
    ppc_poisson = posterior_predictive_array(idata_poisson)

    ppc_plot(data, ppc35[:, :100], "Posterior predictive check: Binomial(n=35)")
    ppc_plot(data, ppc45[:, :100], "Posterior predictive check: Binomial(n=45)")
    ppc_plot(data, ppc_poisson[:, :100], "Posterior predictive check: Poisson")

    summaries = {
        "Binomial(n=35)": ppc_summary(data, ppc35.ravel()),
        "Binomial(n=45)": ppc_summary(data, ppc45.ravel()),
        "Poisson": ppc_summary(data, ppc_poisson.ravel()),
    }

    print("Posterior predictive summaries")
    for name, summary in summaries.items():
        print_ppc_summary(name, summary)

    loo35 = az.loo(idata35, pointwise=True)
    loo45 = az.loo(idata45, pointwise=True)
    loo_poisson = az.loo(idata_poisson, pointwise=True)

    print("LOO comparison; higher elpd_loo is better")
    print(f"  Binomial(n=35): elpd_loo = {loo35.elpd_loo:.2f}, p_loo = {loo35.p_loo:.2f}, se = {loo35.se:.2f}")
    print(f"  Binomial(n=45): elpd_loo = {loo45.elpd_loo:.2f}, p_loo = {loo45.p_loo:.2f}, se = {loo45.se:.2f}")
    print(f"  Poisson:        elpd_loo = {loo_poisson.elpd_loo:.2f}, p_loo = {loo_poisson.p_loo:.2f}, se = {loo_poisson.se:.2f}\n")

    comparison = az.compare(
        {
            "binomial_n35": idata35,
            "binomial_n45": idata45,
            "poisson": idata_poisson,
        },
        method="BB-pseudo-BMA",
        ic="loo",
    )
    print(comparison)


if __name__ == "__main__":
    main()
