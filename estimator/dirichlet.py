import numpy as np
import pandas as pd
from typing import Sequence, Union, Optional

def dirichlet_multinomial_summary(
    counts_df: pd.DataFrame,
    class_cols: Sequence[str],
    alpha: Union[float, Sequence[float]] = 1.0,
    nsim: int = 100_000,
    batch_size: Optional[int] = None,
    random_state: Optional[int] = 42,
) -> pd.DataFrame:
    """
    Vectorized Dirichlet–multinomial summary across many rows (patients).

    Parameters
    ----------
    counts_df : pd.DataFrame
        Must contain integer counts per class in columns `class_cols`.
    class_cols : list[str]
        Column names for the classes, e.g. ["THERAPEUTIC", "SUPRA_THERAPEUTIC", "SUB_THERAPEUTIC"].
    alpha : float or sequence of floats, default=1.0
        Dirichlet prior concentration(s).
        - If scalar, repeats for each class.
        - If sequence, length must equal len(class_cols).
    nsim : int, default=100_000
        Number of Monte Carlo draws to estimate P(class is largest). Increase for tighter Monte Carlo error.
    batch_size : int or None, default=None
        If set, Monte Carlo sampling is split into batches to bound memory.
        Good values: 10_000–50_000 depending on RAM.
    random_state : int or None
        Seed for reproducibility.

    Returns
    -------
    pd.DataFrame with columns:
        - map_class : str
        - prob_largest : float   (posterior probability that map_class is largest)
        - post_mean_<CLASS> : float for each class
    """
    rng = np.random.default_rng(random_state)

    X = counts_df[class_cols].to_numpy(dtype=float)  # shape (N, K)
    if np.any(X < 0):
        raise ValueError("Counts must be non-negative.")

    N, K = X.shape

    # Prior vector
    if np.isscalar(alpha):
        alpha_vec = np.full(K, float(alpha), dtype=float)
    else:
        alpha_vec = np.asarray(alpha, dtype=float)
        if alpha_vec.shape != (K,):
            raise ValueError("alpha must be scalar or length equal to number of classes")

    # Posterior parameters
    alpha_post = X + alpha_vec  # (N, K)
    alpha_post_sum = alpha_post.sum(axis=1, keepdims=True)  # (N, 1)

    # Posterior mean (also MAP class under flat/weak prior)
    post_means = alpha_post / alpha_post_sum  # (N, K)
    map_idx = post_means.argmax(axis=1)       # (N,)

    # Monte Carlo probability that each class is the largest
    # We do batched Gamma sampling to avoid huge memory.
    if batch_size is None:
        batch_size = min(nsim, 50_000)

    # accumulate per-row winner counts
    winners_count = np.zeros((N, K), dtype=np.int64)
    remaining = nsim

    while remaining > 0:
        b = min(batch_size, remaining)
        # Sample Gamma(alpha_post, 1) for shape (b, N, K), then normalize → Dirichlet
        # Vectorized: draw for all rows/classes at once
        # To keep memory modest, we draw per batch.
        gam = rng.gamma(shape=alpha_post, scale=1.0, size=(b, N, K))  # (b, N, K)
        dir_samples = gam / gam.sum(axis=2, keepdims=True)            # (b, N, K)
        # Winners per sample for each row
        # argmax over classes (axis=2) → (b, N)
        winners = dir_samples.argmax(axis=2)
        # Tally winners into winners_count
        # We’ll do this row-by-row for clarity; this loop is over rows only (fast in practice).
        for i in range(N):
            w = np.bincount(winners[:, i], minlength=K)
            winners_count[i, :] += w
        remaining -= b

    # Probability class is largest (per row)
    p_largest = winners_count / nsim  # (N, K)

    # Probability for the chosen map class (confidence)
    prob_map_largest = p_largest[np.arange(N), map_idx]  # (N,)

    # Build output DataFrame
    out = pd.DataFrame(index=counts_df.index)
    out["map_class"] = np.array(class_cols)[map_idx]
    out["prob_largest"] = prob_map_largest

    # Add posterior means per class
    for j, cls in enumerate(class_cols):
        out[f"post_mean_{cls}"] = post_means[:, j]

    return out
