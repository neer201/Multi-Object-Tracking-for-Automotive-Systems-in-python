import time

import numpy as np
import scipy
import scipy.stats


def vectorized_gaussian_logpdf(X, means, covariances):
    """
    Compute log N(x_i; mu_i, sigma_i) for each x_i, mu_i, sigma_i
    https://stackoverflow.com/a/50809036/8395138
    Args:
        X : shape (n, d)
            Data points
        means : shape (n, d)
            Mean vectors
        covariances : shape (n, d)
            Diagonal covariance matrices
    Returns:
        logpdfs : shape (n,)
            Log probabilities
    """
    _, d = X.shape
    constant = d * np.log(2 * np.pi)
    log_determinants = np.log(np.prod(covariances, axis=1))
    deviations = X - means
    inverses = 1 / covariances
    return -0.5 * (constant + log_determinants + np.sum(deviations * inverses * deviations, axis=1))


if __name__ == "__main__":
    n = 128 ** 2
    d = 64

    means = np.random.uniform(-1, 1, (n, d))
    covariances = np.random.uniform(0, 2, (n, d))
    X = np.random.uniform(-1, 1, (n, d))

    refs = []

    ref_start = time.time()
    for x, mean, covariance in zip(X, means, covariances):
        refs.append(scipy.stats.multivariate_normal.logpdf(x, mean, covariance))
    ref_time = time.time() - ref_start

    fast_start = time.time()
    results = vectorized_gaussian_logpdf(X, means, covariances)
    fast_time = time.time() - fast_start

    print("Reference time:", ref_time)
    print("Vectorized time:", fast_time)
    print("Speedup:", ref_time / fast_time)

    assert np.allclose(results, refs)
