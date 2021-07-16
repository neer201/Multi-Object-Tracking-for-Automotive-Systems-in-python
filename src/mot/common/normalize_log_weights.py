from typing import List

import numpy as np


def normalize_log_weights(log_weights: List[float]):
    """Normalize a list of log weights.
    So that sum(exp(log_weights)) = 1

    Args:
        log_weights (List[float]): list of log weights, e.g. log likelihood


    Returns:
        normalized_log_weights (List[float]): log of the normalized weights
        log_sum_w (float): log of the sum of the non-normalized weights
    """

    if len(log_weights) == 1:
        # Compute sum of log weights
        # All weights are unnormilized
        log_sum_w = log_weights[0]
        # Normalize
        normalized_log_weights = log_weights.copy()
        normalized_log_weights[0] -= log_sum_w
    elif len(log_weights) == 0:
        return [], None
    else:
        # Log of sum weights times prior probabilities
        log_weights = np.array(log_weights)
        arg_order = np.argsort(log_weights)  # ascending orgder
        log_sum_w = log_weights[arg_order[-1]]
        log_sum_w += np.log(1 + np.sum(np.exp(log_weights[arg_order[:-1]] - log_weights[arg_order[-1]])))
        normalized_log_weights = (log_weights - log_sum_w).tolist()
    return normalized_log_weights, log_sum_w
