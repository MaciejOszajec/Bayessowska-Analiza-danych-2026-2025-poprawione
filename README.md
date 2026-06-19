# Bayesian Data Analysis — assignments

Repository contains Python solutions for the assignments from the Bayesian Data Analysis course. The programs are written as standalone scripts and use the data files placed in the repository root.

## Contents

| File | Assignment topic | Main elements |
| --- | --- | --- |
| `dice.py` | Dice | Bayesian inference for the probability of rolling a six; uniform prior, informative Beta prior, MAP, mean, median, 90% HDR, crooked dice estimate. |
| `Lighthouse.py` | Lighthouse, one unknown parameter | Grid posterior for `x_lh`, updates after one and two measurements, iterative update for the first 100 measurements, final MAP, 95% symmetric interval, log-probability calculation. |
| `lighthouse2.py` | Lighthouse, two unknown parameters | Grid posterior for `x_lh` and `h`, probability density and log-density functions, posterior after one flash, after two flashes and after all flashes, marginal distributions and MAP estimates. |
| `LighthousMc.py` | Lighthouse, Monte Carlo | PyMC models for the one-dimensional and two-dimensional lighthouse problems, MAP estimates, posterior means and 95% HDI intervals. |
| `ModelSelection.py` | Model selection | Binomial models with `n = 35` and `n = 45`, Poisson model with Gamma prior, posterior predictive checks and LOO comparison. |
| `regression.py` | Quadratic regression | Least-squares fit with `scipy.optimize.curve_fit`, Bayesian fit in PyMC, MAP estimate, posterior sampling, correlation matrix and posterior prediction error. |

## Data files

The root directory contains the data files used directly by the scripts:

- `data.txt`
- `lighthouse.txt`
- `lighthouse_2d.txt`
- `regression.txt`

The default `regression.txt` is the current dataset from the `Assignments/400_Regression` directory. The directory `regression_data/` keeps both regression-data variants:

```text
regression_data/
├── 2025/regression.txt
└── 2026/regression.txt
```

The scripts use only the files in the repository root. To repeat the regression exercise with the older regression dataset, copy `regression_data/2025/regression.txt` over the root-level `regression.txt` and run `regression.py` again.

## Running the scripts

A standard scientific Python environment is sufficient for the grid-based scripts. The PyMC-based scripts additionally require `pymc` and `arviz`.

```bash
pip install -r requirements.txt
```

Examples:

```bash
python dice.py
python Lighthouse.py
python lighthouse2.py
python ModelSelection.py
python LighthousMc.py
python regression.py
```

The programs print numerical summaries to the terminal and display the required diagnostic plots.

## Notes on implementation

- Posterior distributions in the lighthouse grid calculations are normalized by discrete summation over the grid, as required in the assignment.
- Log-probability calculations use `scipy.special.logsumexp` to avoid numerical underflow when many likelihood terms are multiplied.
- The regression model uses wide Gaussian priors for polynomial coefficients and an explicit logarithmic potential for `sigma`, corresponding to the requested prior proportional to `1/sigma` on positive values.
- Posterior predictive checks in the model-selection task are used together with LOO cross-validation, so both visual fit and predictive comparison are reported.
