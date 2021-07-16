from copy import deepcopy
from functools import partial
from typing import List, Tuple

import numpy as np
import scipy

from mot.common import (
    GaussianDensity,
    GaussianMixture,
    Observation,
    ObservationList,
    normalize_log_weights,
)
from mot.measurement_models import MeasurementModel
from mot.motion_models import MotionModel
from mot.utils.timer import Timer

from .bernoulli import Bernoulli
from .single_target_hypothesis import SingleTargetHypothesis
from .track import Track


class PoissonRFS:
    def __init__(self, intensity: GaussianMixture):
        assert isinstance(intensity, GaussianMixture)
        self.intensity = deepcopy(intensity)
        # self.pool = Pool(nodes=8)

    def __repr__(self):
        return self.intensity.__repr__()

    def __len__(self):
        return len(self.intensity)

    def get_targets_detected_for_first_time(
        self,
        measurements: ObservationList,
        clutter_intensity: float,
        meas_model: MeasurementModel,
        detection_probability: float,
    ) -> List[Track]:

        H_x, S, K = GaussianDensity.numpy_get_Kalman_gain(self.intensity, meas_model)
        # new_single_target_hypotheses = [
        #     self.detected_update(
        #         meas_idx=meas_idx,
        #         measurement=measurements[meas_idx],
        #         meas_model=meas_model,
        #         detection_probability=detection_probability,
        #         clutter_intensity=clutter_intensity,
        #     ) for meas_idx in range(len(measurements))
        # ]

        detected_update_func = partial(
            PoissonRFS.detected_update,
            intensity=self.intensity,
            meas_model=meas_model,
            detection_probability=detection_probability,
            clutter_intensity=clutter_intensity,
        )

        # new_single_target_hypotheses = self.pool.map(
        #     detected_update_func,
        #     [(idx, measurements[idx]) for idx in range(len(measurements))])

        new_single_target_hypotheses = [
            detected_update_func((meas_idx, measurements[meas_idx])) for meas_idx in range(len(measurements))
        ]
        # new_single_target_hypotheses = Parallel(
        #     n_jobs=-1)(delayed(self.detected_update)(
        #         meas_idx=meas_idx,
        #         measurement=measurements[meas_idx],
        #         meas_model=meas_model,
        #         detection_probability=detection_probability,
        #         clutter_intensity=clutter_intensity)
        #                         for meas_idx in range(len(measurements)))

        new_tracks = {}
        for meas_idx in range(len(measurements)):
            new_track = Track.from_sth(new_single_target_hypotheses[meas_idx])
            new_tracks[new_track.track_id] = new_track
        return new_tracks

    @staticmethod
    def detected_update(
        meas: Tuple[int, Observation],
        intensity,
        meas_model: MeasurementModel,
        detection_probability: float,
        clutter_intensity: float,
        density=GaussianDensity,
    ) -> SingleTargetHypothesis:
        """Creates a new local hypothesis by updating the PPP
        with measurement and calculates the corresponding
        likelihood.

        z np .ndarray
            number of measurementx x measurement dimension

        Resulting Bernoulli component represents two possibilites:
        1) deteciton from clutter
        2) detection from a new object

        NOTE: p.152 in presentation

        """
        meas_idx, measurement = meas
        assert isinstance(meas_model, MeasurementModel)
        # 1. For each mixture component in the PPP intensity, perform Kalman update and
        # calculate the predicted likelihood for each detection inside the corresponding ellipsoidal gate.

        # TODO remove redundant deep copy - it takes 0.012 secs
        (
            updated_ppp_components,
            loglikelihoods,
        ) = GaussianDensity.update_states_with_likelihoods_by_single_measurement(
            intensity, measurement.measurement, meas_model
        )

        # references = [
        #     GaussianDensity.update(state.gaussian, measurement, meas_model)
        #     for state in copy.deepcopy(self.intensity)
        # ]

        # Compute predicted likelihood
        assert len(intensity) == len(loglikelihoods)
        log_weights = np.array(
            [
                np.log(detection_probability) + ppp_component.log_weight + loglikelihood
                for ppp_component, loglikelihood in zip(intensity, loglikelihoods)
            ]
        )

        # 2. Perform Gaussian moment matching for the updated object state densities
        # resulted from being updated by the same detection.
        normalized_log_weights, log_sum = normalize_log_weights(log_weights)
        merged_state = density.moment_matching_vectorized(normalized_log_weights, updated_ppp_components)

        # 3. The returned likelihood should be the sum of the predicted likelihoods calculated f
        # or each mixture component in the PPP intensity and the clutter intensity.
        # (You can make use of the normalizeLogWeights function to achieve this.)
        log_likelihood = scipy.special.logsumexp([log_sum, np.log(clutter_intensity)])

        # 4. The returned existence probability of the Bernoulli component
        # is the ratio between the sum of the predicted likelihoods
        # and the returned likelihood.
        # (Be careful that the returned existence probability is
        # in decimal scale while the likelihoods you calculated
        # beforehand are in logarithmic scale)
        existence_probability = np.exp(log_sum - log_likelihood)
        bernoulli = Bernoulli(merged_state, existence_probability, measurement.metadata)
        cost = -log_likelihood
        return SingleTargetHypothesis(
            bernoulli=bernoulli,
            log_likelihood=log_likelihood,
            cost=cost,
            meas_idx=meas_idx,
            sth_id=0,
        )

    @Timer(name="update ppp componentns for missed detetion")
    def undetected_update(self, detection_probability) -> None:
        """Performs PPP update for missed detection."""
        for ppp_component in self.intensity:
            ppp_component.log_weight += np.log(1 - detection_probability)

    def prune(self, threshold: float) -> None:
        self.intensity.weighted_components = [
            ppp_component for ppp_component in self.intensity if ppp_component.log_weight > threshold
        ]

    def predict(
        self,
        motion_model: MotionModel,
        survival_probability: float,
        density: GaussianDensity,
        dt: float,
    ) -> None:
        """Performs prediciton step for PPP components hypothesing undetected objects.
        Birth components will be added in another method."""
        assert isinstance(motion_model, MotionModel)
        assert isinstance(survival_probability, float)
        assert isinstance(dt, float)

        for ppp_component in self.intensity:
            ppp_component.log_weight += np.log(survival_probability)
            ppp_component.gaussian = density.predict(ppp_component.gaussian, motion_model, dt)

    def birth(self, new_components: GaussianMixture):
        """Incorporate PPP birth intensity into PPP intensity

        Parameters
        ----------
        born_components : GaussianMixture
            [description]
        """
        assert isinstance(new_components, GaussianMixture)
        self.intensity.extend(deepcopy(new_components))

    def gating(
        self,
        measurements,
        density_handler,
        meas_model: MeasurementModel,
        gating_size: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Returns measurement indices inside the gate of undetected objects (PPP)"""
        gating_matrix_undetected = np.full(
            shape=[self.intensity.size, len(measurements)], fill_value=False
        )  # poisson size x number of measurements
        used_measurement_undetected_indices = np.full(shape=[len(measurements)], fill_value=False)

        for ppp_idx in range(self.intensity.size):
            _, gating_matrix_undetected[ppp_idx, :] = density_handler.ellipsoidal_gating(
                self.intensity[ppp_idx].gaussian, measurements, meas_model, gating_size
            )
            used_measurement_undetected_indices = np.logical_or(
                used_measurement_undetected_indices, gating_matrix_undetected[ppp_idx]
            )
        return (gating_matrix_undetected, used_measurement_undetected_indices)
