import numpy as np
import pytest
from mot.common.state import GaussianMixture, WeightedGaussian, Gaussian


@pytest.fixture(scope="function")
def initial_PPP_intensity_linear():
    return GaussianMixture(
        [
            WeightedGaussian(
                np.log(0.03),
                Gaussian(x=np.array([-400.0, 200.0, 0.0, 0.0]), P=400 * np.eye(4)),
            ),
            WeightedGaussian(
                np.log(0.03),
                Gaussian(x=np.array([-400.0, -200.0, 0.0, 0.0]), P=400 * np.eye(4)),
            ),
        ]
    )


@pytest.fixture(scope="function")
def initial_PPP_intensity_nonlinear():
    return GaussianMixture(
        [
            WeightedGaussian(
                -0.3861,
                Gaussian(x=np.array([0.0, 0.0, 5.0, 0.0, np.pi / 180]), P=np.eye(5)),
            ),
            WeightedGaussian(
                -0.423,
                Gaussian(x=np.array([20.0, 20.0, -20.0, 0.0, np.pi / 90]), P=np.eye(5)),
            ),
            WeightedGaussian(
                -1.8164,
                Gaussian(x=np.array([-20.0, 10.0, -10.0, 0.0, np.pi / 360]), P=np.eye(5)),
            ),
        ]
    )
