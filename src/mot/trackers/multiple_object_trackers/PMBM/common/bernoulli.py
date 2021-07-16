import numpy as np

from mot.common import Gaussian, GaussianDensity, ObjectMetadata
from mot.measurement_models import MeasurementModel
from mot.motion_models import MotionModel


class Bernoulli:
    """Bernoulli component

    Parameters
    ----------
    state : Gaussian
        a struct contains parameters describing the object pdf
    r : scalar
        probability of existence
    """

    def __init__(self, state: Gaussian, existence_probability: float, metadata: ObjectMetadata = None):
        self.state: Gaussian = state
        self.existence_probability: float = existence_probability
        self.metadata = metadata

    def __repr__(self) -> str:
        return self.__class__.__name__ + (f"(r={self.existence_probability:.4f}, " f"state={self.state}")

    def predict(
        self,
        motion_model: MotionModel,
        survival_probability: float,
        density: GaussianDensity,
        dt: float = 1.0,
    ) -> None:
        """Performs prediciton step for a Bernoulli component"""

        # Probability of survival * Probability of existence
        self.existence_probability = survival_probability * self.existence_probability

        # Kalman prediction of the new state
        self.state = density.predict(self.state, motion_model, dt)
        return self

    def undetected_update_state(self, detection_probability: float):
        """Calculates the likelihood of missed detection,
        and creates new local hypotheses due to missed detection.
        NOTE: from page 88 lecture 04
        """

        posterior_existence_probability = (self.existence_probability * (1 - detection_probability)) / (
            1 - self.existence_probability + self.existence_probability * (1 - detection_probability)
        )
        posterior_bern = Bernoulli(self.state, posterior_existence_probability)
        return posterior_bern

    def undetected_update_loglikelihood(self, detection_probability: float):
        missdetecion_probability = 1 - detection_probability
        likelihood_predicted = 1 - self.existence_probability + self.existence_probability * missdetecion_probability
        log_likelihood_predicted = np.log(likelihood_predicted)
        return log_likelihood_predicted

    def detected_update_loglikelihood(
        self,
        measurement: np.ndarray,
        meas_model: MeasurementModel,
        detection_probability: float,
        density=GaussianDensity,
    ) -> float:
        """Calculates the predicted likelihood for a given local hypothesis.
        NOTE page 86 lecture 04
        """

        assert isinstance(meas_model, MeasurementModel)

        log_likelihood_detected = (
            density.predict_loglikelihood(self.state, measurement, meas_model)
            + np.log(detection_probability)
            + np.log(self.existence_probability)
        )
        return log_likelihood_detected

    def detected_update_state(
        self,
        measurement: np.ndarray,
        meas_model: MeasurementModel,
        density=GaussianDensity,
    ):
        """Creates the new local hypothesis due to measurement update.
        NOTE: page 85 lecture 04

        """
        assert isinstance(meas_model, MeasurementModel)

        updated_density = density.update(self.state, measurement, meas_model)
        update_bern = Bernoulli(state=updated_density, existence_probability=1.0)
        return update_bern
