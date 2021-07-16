import numpy as np
import scipy


def GOSPA(
    targets: np.ndarray,
    estimates: np.ndarray,
    p: float = 1,
    c: float = 100,
    alpha: float = 2,
    state_dim: int = 2,
) -> float:
    """GOSPA - Generalized optimal sub-pattern assignment metric
    TODO: add paper and formula

    Parameters
    ----------
    targets : np.array (num of states x state dim)
        Describes ground truth states of objects. Order is not important.
    estimates : np.array (num of states x state dim)
        Describes object state estimation provided usually by filter. Order is not important.
    p : float
        Order parameter. A high value penalizes outliers more.
    c : float
        The maximum allowable localization error, and also determines the cost of a cardinality missmatch.
    alpha : float, optional
        Defines the cost of a missing target or false estimate along with c.
        The default value is 2, which is the most suited value for tracking algorithms.
    state_dim : int
        [description]

    Returns
    -------
    float
        GOSPA metric
    """
    # eval position x and position y without velocities

    targets_number = targets.shape[0]
    estimates_number = estimates.shape[0]

    if targets_number > estimates_number:
        return GOSPA(estimates, targets, p, c, alpha, state_dim)

    elif targets_number == 0:
        return c ** p / alpha * estimates_number

    costs = scipy.spatial.distance.cdist(targets, estimates)
    costs = np.minimum(costs, c) ** p

    row_ind, col_ind = scipy.optimize.linear_sum_assignment(costs)

    gospa_scalar = np.sum(costs[row_ind, col_ind]) + c ** p / alpha * (estimates_number - targets_number)

    return gospa_scalar
