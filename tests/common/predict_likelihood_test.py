import numpy as np

from mot.common.gaussian_density import GaussianDensity
from mot.common.state import Gaussian
from mot.measurement_models import ConstantVelocityMeasurementModel


TOL = 1e-4


def test_predict_likelihood_long_distance():
    state = Gaussian(x=np.array([20, 20, 10, 10]), P=np.eye(4))
    meas_model = ConstantVelocityMeasurementModel(5.0)
    measurement = np.array([-100, -100]).reshape([1, 2])
    predicted_likelihood = GaussianDensity.predict_loglikelihood(
        state,
        z=measurement,
        measurement_model=meas_model,
    )
    np.testing.assert_allclose(predicted_likelihood, -558.9421, rtol=0.01)


def test_predict_likelihood_shore_distance():
    state = Gaussian(x=np.array([20, 20, 10, 10]), P=np.eye(4))
    measurement = np.array([11, 11]).reshape([1, 2])
    meas_model = ConstantVelocityMeasurementModel(5.0)
    predicted_likelihood = GaussianDensity.predict_loglikelihood(
        state,
        z=measurement,
        measurement_model=meas_model,
    )
    np.testing.assert_allclose(predicted_likelihood, -8.2114, rtol=0.01)
