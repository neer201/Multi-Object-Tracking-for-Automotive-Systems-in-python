import numpy as np

from mot.common import Gaussian, GaussianMixture, WeightedGaussian


birth_model_params = GaussianMixture(
    [
        WeightedGaussian(
            np.log(0.03),
            Gaussian(x=np.array([0.0, 0.0, 0.0, 0.0]), P=100 * np.eye(4)),
        ),
        WeightedGaussian(
            np.log(0.03),
            Gaussian(x=np.array([400.0, -600.0, 0.0, 0.0]), P=100 * np.eye(4)),
        ),
        WeightedGaussian(
            np.log(0.03),
            Gaussian(x=np.array([-800.0, 200.0, 0.0, 0.0]), P=100 * np.eye(4)),
        ),
        WeightedGaussian(
            np.log(0.03),
            Gaussian(x=np.array([-200.0, 800.0, 0.0, 0.0]), P=100 * np.eye(4)),
        ),
    ]
)
