import numpy as np
import scipy
import scipy.io

from mot.common.gaussian_density import GaussianDensity
from mot.common.state import Gaussian


TOL = 1e-4


def test_moment_matching_small():
    # TODO
    """
        w =

        1.0000
        0.0000

        K>> states.x

    ans =

       24.3498
        5.7689
        5.0000
             0
        0.0175


    ans =

        4.7970
       14.7449
      -10.0000
             0
        0.0087

        P
        ans =

        0.9779   -0.0122         0         0         0
       -0.0122    0.9707         0         0         0
             0         0    1.0000         0         0
             0         0         0    1.0000         0
             0         0         0         0    1.0000


    ans =

        0.9769   -0.0126         0         0         0
       -0.0126    0.9719         0         0         0
             0         0    1.0000         0         0
             0         0         0    1.0000         0
             0         0         0         0    1.0000
    """
    raise NotImplementedError


def test_moment_matching_big():
    num_gaussians = 5  # noqa F841
    n_dim = 4  # noqa F841
    states = []  # noqa F841
    raise NotImplementedError
    expected_vars = scipy.io.loadmat("tests/data/SA2Ex2Test3.mat")

    test_w = np.array(expected_vars["w"]).squeeze()
    test_states = [
        Gaussian(x=np.array(test_x[0]).squeeze(), P=np.array(test_P)[0])
        for test_x, test_P in zip(expected_vars["states"]["x"], expected_vars["states"]["P"])
    ]
    got_state = GaussianDensity.moment_matching(test_w, test_states)
    expected_state = Gaussian(
        x=expected_vars["state_ref"]["x"][0][0].squeeze(),
        P=expected_vars["state_ref"]["P"][0][0],
    )
    assert np.linalg.norm(got_state.x).all() > TOL, "check calculation of mean"
    assert np.linalg.norm(got_state.P - expected_state.P).all() > TOL, "check calculation of covariance"
