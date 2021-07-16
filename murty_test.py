import copy
from dataclasses import dataclass

import numpy as np
from murty import Murty
from scipy.optimize import linear_sum_assignment
from slowmurty import mhtda as mhtdaSlow


inf = np.inf


@dataclass
class MurtyPair:
    row: int
    col: int

    def is_solution(self):
        if self.row in [-2]:
            return False
        if self.col in [-2]:
            return False
        return True


def convert_murty_output(murty_assignment, n_rows):

    murty_solver = Murty(copy.deepcopy(cost_matrix))  # noqa F841
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    murty_sol = [MurtyPair(sol[0], sol[1]) for sol in murty_assignment]
    murty_sol = [murty_pair for murty_pair in murty_sol if murty_pair.is_solution()]
    return murty_sol
    # solution = np.zeros(nrows)
    # for murty_pair in murty_sol:
    #     solution[murty_pair.row] = murty_pair.col
    # return solution


if __name__ == "__main__":
    cost_matrix = np.loadtxt("./bad_cost_matrices/cost_matrix_2.csv", delimiter=",")

    # cost_matrix = np.array([[-10, -9, -1], [-1, -6, 3], [-9, -5, -6]],
    #                        dtype=np.float64)
    nrows, ncolumns = cost_matrix.shape
    nsolutions = 6  # find the 5 lowest-cost associations

    # sparse cost matrices only include a certain number of elements
    # the rest are implicitly infinity
    # in this case, the sparse matrix includes all elements (3 columns per row)

    # mhtda is set up to potentially take multiple input hypotheses for both rows and columns
    # input hypotheses specify a subset of rows or columns.
    # In this case, we just want to use the whole matrix.
    row_priors = np.ones((1, nrows), dtype=np.bool8)
    col_priors = np.ones((1, ncolumns), dtype=np.bool8)
    # Each hypothesis has a relative weight too.
    # These values don't matter if there is only one hypothesis...
    row_prior_weights = np.zeros(1)
    col_prior_weights = np.zeros(1)

    # The mhtda function modifies preallocated outputs rather than
    # allocating new ones. This is slightly more efficient for repeated use
    # within a tracker.
    # The cost of each returned association:
    out_costs = np.zeros(nsolutions)
    # The row-column pairs in each association:
    # Generally there will be less than nrows+ncolumns pairs in an association.
    # The unused pairs are currently set to (-2, -2)
    out_associations = np.zeros((nsolutions, nrows + ncolumns, 2), dtype=np.int32)
    mhtdaSlow(cost_matrix, row_priors, row_prior_weights, col_priors, col_prior_weights, out_associations, out_costs)

    # associations = []
    converted_associations = [convert_murty_output(association, nrows) for association in out_associations]
    import pdb

    pdb.set_trace()
