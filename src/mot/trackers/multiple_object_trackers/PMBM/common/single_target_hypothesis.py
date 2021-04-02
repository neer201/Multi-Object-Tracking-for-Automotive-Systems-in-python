import numpy as np
from mot.common.gaussian_density import GaussianDensity
from mot.measurement_models.base_measurement_model import MeasurementModel
from .bernoulli import Bernoulli


class SingleTargetHypothesis:
    def __init__(
        self,
        bernoulli: Bernoulli,
        log_likelihood: float,
        cost: float = None,
        meas_idx: int = None,
        sth_id: int = None,
    ):
        assert isinstance(bernoulli, Bernoulli)
        self.bernoulli = bernoulli
        self.log_likelihood = log_likelihood
        self.cost = cost
        self.meas_idx = meas_idx  # associated measurements
        self.sth_id = sth_id
        self.missdetection_hypothesis = None
        self.detection_hypotheses = {}

    def __repr__(self):
        return f"likelihood={self.log_likelihood:.2f}, " f"meas_idx={self.meas_idx}, "

    def create_missdetection_hypothesis(self, detection_probability: float, sth_id):
        missdetection_bernoulli = self.bernoulli.undetected_update_state(
            detection_probability
        )
        missdetection_loglikelihood = self.bernoulli.undetected_update_loglikelihood(
            detection_probability
        )
        return SingleTargetHypothesis(
            bernoulli=missdetection_bernoulli,
            log_likelihood=missdetection_loglikelihood,
            sth_id=sth_id,
        )

    def create_detection_hypothesis(
        self,
        measurement: np.ndarray,
        detection_probability: float,
        meas_model: MeasurementModel,
        density: GaussianDensity,
        sth_id: int,
    ):
        assert measurement.ndim == 1
        detection_bernoulli = self.bernoulli.detected_update_state(
            measurement, meas_model, density
        )
        detection_log_likelihood = self.bernoulli.detected_update_loglikelihood(
            measurement, meas_model, detection_probability, density
        )
        missdetection_log_likelihood = (
            self.missdetection_hypothesis.log_likelihood
            or self.bernoulli.undetected_update_loglikelihood(detection_probability)
        )
        detection_hypothesis = SingleTargetHypothesis(
            bernoulli=detection_bernoulli,
            log_likelihood=detection_log_likelihood,
            cost=-(detection_log_likelihood - missdetection_log_likelihood),
            sth_id=sth_id,
        )
        return detection_hypothesis