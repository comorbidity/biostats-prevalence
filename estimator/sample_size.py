import json
import pandas as pd
import numpy as np
from scipy.stats import norm

def n_single_prevalence(prevalence: float, margin_error: float = 0.05, alpha: float = 0.05, fpc_N: int | None = None, deff: float = 1.0) -> float:
    """
    Calculate the required sample size (n) for estimating a single population prevalence
    with a specified margin of error and confidence level.

    Parameters
    ----------
    prevalence : float
        Expected prevalence (proportion between 0 and 1).
    margin_error : float
        Desired margin of error (absolute precision of the estimate, e.g., 0.05 for ±5%).
    alpha : float, optional
        Significance level for the confidence interval (default is 0.05 for 95% confidence).
    fpc_N : int or None, optional
        Finite population correction (FPC) — total population size `N`.
        If provided, applies the correction factor to reduce sample size when the population is small.
    deff : float, optional
        Design effect (default = 1.0).
        Adjusts the sample size for complex survey designs such as clustering or stratification.

    Returns
    -------
    float
        Required sample size `n` for the given parameters.

    Notes
    -----
    - The calculation uses the standard normal approximation:
      n = (z^2 * p * (1 - p)) / (margin_error^2)
    - The finite population correction (if `fpc_N` is provided) is applied as:
      n_corrected = (n * N) / (n + N - 1)
    - The z-value is obtained from the two-tailed normal quantile for `1 - alpha/2`.
    """
    z = norm.ppf(1 - alpha/2.0)
    n = (z**2) * prevalence * (1 - prevalence) / (margin_error ** 2)
    n *= deff
    if fpc_N is not None:
        n = (n * fpc_N) / (n + fpc_N - 1)
    return n

def n_two_proportions(p1: float, p2: float, alpha: float = 0.05, power: float = 0.80, ratio: float = 1.0, deff: float = 1.0) -> tuple[float, float]:
    z_alpha = norm.ppf(1 - alpha/2.0)
    z_beta  = norm.ppf(power)
    k = ratio
    p_bar = (p1 + k*p2) / (1 + k)
    q_bar = 1 - p_bar
    se0 = np.sqrt((1 + 1/k) * p_bar * q_bar)
    se1 = np.sqrt(p1*(1-p1) + (p2*(1-p2))/k)
    delta = abs(p1 - p2)
    n1 = ((z_alpha*se0 + z_beta*se1) / delta) ** 2
    n1 *= deff
    n2 = k * n1
    return n1, n2


if __name__ == '__main__':
    print(n_single_prevalence(0.8))
