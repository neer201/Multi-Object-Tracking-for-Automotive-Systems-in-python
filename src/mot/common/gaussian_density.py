import copy
import logging
from typing import List

import numpy as np
import numpy.typing as npt
import scipy.stats  # noqa: I201

from mot.common.normalize_log_weights import normalize_log_weights
from mot.common.state import Gaussian, GaussianMixture
from mot.measurement_models import MeasurementModel
from mot.motion_models import MotionModel
from mot.utils import vectorized_gaussian_logpdf


class GaussianDensity:
    def __init__(self, parameter_list):
        """
        docstring
        """

    @staticmethod
    def predict(state: Gaussian, motion_model: MotionModel, dt: float) -> Gaussian:
        """Performs linear/nonlinear (Extended) Kalman prediction step

        Args:
            state (Gaussian): a structure with two fields - mean and covariance
            motion_model (MotionModel): a structure specifies the motion model parameters

        Returns:
            state_pred (Gaussian): predicted value of state
        """
        next_x = motion_model.f(state.x, dt)
        next_F = motion_model.F(state.x, dt)
        next_P = np.linalg.multi_dot([next_F, state.P, next_F.T]) + motion_model.Q(dt)
        return Gaussian(next_x, next_P)

    @staticmethod
    def update_states_with_likelihoods_by_single_measurement(
        initial_states: GaussianMixture,
        measurement: np.ndarray,
        measurement_model: MeasurementModel,
    ):

        H_x = measurement_model.H(initial_states.states_np)
        # Innovation covariance
        S = H_x @ initial_states.covariances_np @ H_x.T + measurement_model.R

        # Make sure matrix S is positive definite
        S = 0.5 * (S + np.transpose(S, axes=(0, 2, 1)))

        K = initial_states.covariances_np @ H_x.T @ np.linalg.inv(S)

        measurement_row = np.vstack([measurement] * initial_states.size)
        fraction = measurement_row - measurement_model.h(initial_states.states_np.T).T
        with_K = np.einsum("ijk,ik->ij", K, fraction)
        new_states = initial_states.states_np + with_K

        state_vector_size = initial_states.states_np[0].shape[0]

        next_covariances = (np.eye(state_vector_size) - K @ H_x) @ initial_states.covariances_np

        next_states = [Gaussian(new_states[idx], next_covariances[idx]) for idx in range(initial_states.size)]

        measurements_bar = np.expand_dims(H_x, axis=0) @ initial_states.states_np.T

        # it takes 0.0277 sec
        # TODO: clean up
        # loglikelihoods = [
        #     multivariate_normal.logpdf(measurement, measurements_bar[0].T[idx], S[idx])
        #     for idx in range(initial_states.size)
        # ]

        loglikelihoods_fast = vectorized_gaussian_logpdf(
            X=measurement_row,
            means=measurements_bar.squeeze().T,
            covariances=np.diagonal(S, axis1=2),
        )

        # np.testing.assert_almost_equal(loglikelihoods, loglikelihoods_fast)

        return next_states, loglikelihoods_fast

    @staticmethod
    def numpy_get_Kalman_gain(initial_states: GaussianMixture, measurement_model: MeasurementModel):
        H_x = measurement_model.H(initial_states["gaussian"]["means"])

        # Innovation covariance
        states_dot_H_x = initial_states["gaussian"]["covariances"] @ H_x.T
        S = H_x @ states_dot_H_x + measurement_model.R

        # Make sure matrix S is positive definite
        S = 0.5 * (S + np.transpose(S, axes=(0, 2, 1)))

        K = states_dot_H_x @ np.linalg.inv(S)

        return H_x, S, K

    @staticmethod
    def numpy_update_states_with_likelihoods_by_single_measurement(
        initial_states: GaussianMixture,
        measurement: np.ndarray,
        measurement_model: MeasurementModel,
    ):
        H_x = measurement_model.H(initial_states["gaussian"]["means"])

        # Innovation covariance
        states_dot_H_x = initial_states["gaussian"]["covariances"] @ H_x.T
        S = H_x @ states_dot_H_x + measurement_model.R

        # Make sure matrix S is positive definite
        S = 0.5 * (S + np.transpose(S, axes=(0, 2, 1)))

        K = states_dot_H_x @ np.linalg.inv(S)

        measurement_row = np.tile(measurement, (initial_states.shape[0], 1))
        fraction = measurement_row - measurement_model.h(initial_states["gaussian"]["means"].T).T
        fraction_with_K = np.einsum("ijk,ik->ij", K, fraction)

        new_states = initial_states["gaussian"]["means"] + fraction_with_K

        state_vector_size = initial_states["gaussian"]["means"][0].shape[0]

        next_covariances = (np.eye(state_vector_size) - K @ H_x) @ initial_states["gaussian"]["covariances"]

        next_states = np.zeros(initial_states.size, dtype=initial_states["gaussian"].dtype)
        next_states["means"], next_states["covariances"] = new_states, next_covariances
        measurements_bar = np.expand_dims(H_x, axis=0) @ initial_states["gaussian"]["means"].T

        # it takes 0.0277 sec
        # TODO: clean up
        # loglikelihoods = [
        #     multivariate_normal.logpdf(measurement, measurements_bar[0].T[idx], S[idx])
        #     for idx in range(initial_states.size)
        # ]

        loglikelihoods_fast = vectorized_gaussian_logpdf(
            X=measurement_row,
            means=measurements_bar.squeeze().T,
            covariances=np.diagonal(S, axis1=2),
        )

        # np.testing.assert_almost_equal(loglikelihoods, loglikelihoods_fast)
        return next_states, loglikelihoods_fast

    @staticmethod
    def update_state_by_multiple_measurement(
        initial_state: GaussianMixture,
        measurements: np.ndarray,
        measurement_model: MeasurementModel,
    ):
        H_x = measurement_model.H(initial_state.x)
        # Innovation covariance

        S = H_x @ initial_state.P @ H_x.T + measurement_model.R

        # Make sure matrix S is positive definite
        S = 0.5 * (S + np.transpose(S))

        K = initial_state.P @ H_x.T @ np.linalg.inv(S)

        measurement_row = measurements
        fraction = measurement_row - measurement_model.h(initial_state.x.T).T

        with_K = K @ fraction.T

        next_states = np.vstack([initial_state.x] * len(measurements)) + with_K.T

        state_vector_size = initial_state.x.shape[0]

        next_covariances = (np.eye(state_vector_size) - K @ H_x) @ initial_state.P

        return next_states, next_covariances

    @staticmethod
    def update_likelihoods_vectorized(
        updated_states,
        updated_covariances,
        measurements: np.ndarray,
        measurement_model: MeasurementModel,
    ):
        H_x = measurement_model.H(updated_states)
        # Innovation covariance

        S = H_x @ updated_covariances @ H_x.T + measurement_model.R

        # Make sure matrix S is positive definite
        S = 0.5 * (S + np.transpose(S))

        # it takes 0.0277 sec
        # # TODO: clean up
        bar = H_x @ updated_states.T

        # loglikelihoods = [
        #     scipy.stats.multivariate_normal.logpdf(measurements[idx], bar.T[idx], S)
        #     for idx in range(len(measurements))
        # ]

        loglikelihoods = vectorized_gaussian_logpdf(
            X=measurements,
            means=bar.T,
            covariances=np.vstack([np.diagonal(S)] * len(measurements)),
        )
        return loglikelihoods

    @staticmethod
    def update(state_pred: Gaussian, z, measurement_model: MeasurementModel) -> Gaussian:
        """
        Performs linear/nonlinear (Extended) Kalman update step

        Args:
            z (Gaussian): a structure with two fields - mean and covariance
            measurement_model (MeasurementModel): a structure specifies
                                                  the measurement model parameters

        Returns:

            state_upd (Gaussian): updated state
        """
        if z.size == 0:
            logging.error("z size 0")
            return state_pred
        assert isinstance(state_pred, Gaussian)
        if z.ndim == 1:
            z = np.expand_dims(z, axis=0)

        assert z.shape[1] == measurement_model.dim

        # Measurement model Jacobian
        H_x = measurement_model.H(state_pred.x)

        # Innovation covariance
        S = np.linalg.multi_dot([H_x, state_pred.P, H_x.T]) + measurement_model.R

        # Make sure matrix S is positive definite
        S = 0.5 * (S + S.T)

        K = state_pred.P @ H_x.T @ np.linalg.inv(S)

        next_x = state_pred.x + (K @ (z - measurement_model.h(state_pred.x)).T).squeeze()

        assert next_x.shape == state_pred.x.shape

        # Covariance update
        # TODO ndim property for state
        I = np.eye(state_pred.x.shape[0])  # noqa: E741
        next_P = (I - K @ H_x) @ state_pred.P

        state_upd = Gaussian(x=next_x, P=next_P)
        return state_upd

    @staticmethod
    def predict_loglikelihood(
        state_pred: Gaussian, z: npt.ArrayLike, measurement_model: MeasurementModel
    ) -> np.ndarray:
        """Calculates the predicted likelihood in logarithm domain
        Returns:
            predicted_loglikelihood (np.ndarray (number of measurements)): predicted likelihood
                                                                        for each measurement
        """

        assert z.ndim == 2
        assert z.shape[1] == measurement_model.dim
        # Measurement model Jacobian (z_bar)
        H_x = measurement_model.H(state_pred.x)
        # Innovation covariance
        S = H_x @ state_pred.P @ H_x.T + measurement_model.R
        # Make sure matrix S is positive definite
        S = (S + S.T) / 2

        # Predicted measurement
        z_bar = H_x @ state_pred.x

        predicted_loglikelihood = np.zeros(z.shape[0])  # size - num of measurements

        for idx in range(z.shape[0]):
            predicted_loglikelihood[idx] = scipy.stats.multivariate_normal.logpdf(z[idx], z_bar, S)

        assert predicted_loglikelihood.shape[0] == z.shape[0]
        return predicted_loglikelihood

    @staticmethod
    def ellipsoidal_gating(
        state_prev: Gaussian,
        z: npt.ArrayLike,
        measurement_model: MeasurementModel,
        gating_size: float,
    ) -> np.ndarray:
        """Performs ellipsoidal gating for a single object

        Args:
            state_pred (Gaussian): predicted state
            z (np.ndarray (measurements dimenstion) x (number of measurements)): [description]
            measurement_model (MeasurementModel): specifies the measurement model parameters
            gating_size (float): gating size

        Returns:
            z_ingate (np.ndarray (measuremens dim) x (num of measurements in the gate))
                                                         : measurements in the gate
            meas_in_gate (List (num of measurements x 1)): boolean vector indicating whether
                                                           the corresponding measurement is
                                                           in the gate or not
        """
        if z.size == 0:
            logging.warning("No measurements! No updates!")
            return np.array([]), None
        assert z.shape[1] == measurement_model.dim

        # Measurements model Jacobian
        H_x = measurement_model.H(state_prev.x)

        # Innovation covariance
        S = H_x @ state_prev.P @ H_x.T

        # Make sure matrix S is positive definite
        S = (S + S.T) / 2

        # Predicted measurement
        z_bar = H_x @ state_prev.x

        # Difference between measurement and prediction
        z_diff = z - z_bar

        num_of_measurements = z.shape[0]

        # Squared Mahalanobis distance
        Machlanobis_dist = np.zeros(num_of_measurements)

        # TODO vectorize this
        for i in range(num_of_measurements):
            try:
                Machlanobis_dist[i] = z_diff[i] @ np.linalg.inv(S) @ z_diff[i].T
            except np.linalg.LinAlgError:
                logging.warning("It seems, cannot inverse S, skip step")
                return np.array([]), []
        # Machlanobis_dist = mln(z, z_bar, S) # z_diff @ np.linalg.inv(S) @ z_diff.Tx

        indices_in_gate = Machlanobis_dist < gating_size
        assert Machlanobis_dist.shape[0] == z.shape[0]
        z_ingate = z[indices_in_gate]
        return z_ingate, indices_in_gate

    @staticmethod
    def moment_matching(weights: List[float], states: List[Gaussian]) -> Gaussian:
        """Aproximates a Gaussian mixture density as a single Gaussian using moment matching

        Args:
            weights (List(float)): normalized weight of Gaussian components
                                   in logarithm domain
            states (List(Gaussian)): list of Gaussians

        Returns:
            Gaussian: resulted mixture
        """
        if len(weights) == 0:
            return

        log_weights = np.exp(weights)

        # Mean of the states
        N = len(weights)
        x_bar = np.zeros(states[0].x.shape[0])

        for state, weight in zip(states, log_weights):
            x_bar += weight * state.x

        # Covariance
        P_bar = np.zeros_like(states[0].P)
        for idx in range(N):
            d = x_bar - states[idx].x
            P_bar += (states[idx].P + d @ d.T) * log_weights[idx]

        matched_state = Gaussian(x=x_bar, P=P_bar)
        return matched_state

    @staticmethod
    def moment_matching_vectorized(log_weights: List[float], states: List[Gaussian]) -> Gaussian:
        """Aproximates a Gaussian mixture density as a single Gaussian using moment matching

        Args:
            weights (List(float)): normalized weight of Gaussian components
                                   in logarithm domain
            states (List(Gaussian)): list of Gaussians

        Returns:
            Gaussian: resulted mixture
        """
        if len(log_weights) == 0:
            return

        weights = np.exp(log_weights)

        x_bar_np = np.average(np.array([state.x for state in states]), axis=0, weights=weights)
        delta_state = x_bar_np - np.array([state.x for state in states])
        ps = np.array([state.P for state in states]).T + np.einsum("ij,ij->i", delta_state, delta_state)

        P_bar_np = np.average(ps.T, axis=0, weights=weights)
        matched_state = Gaussian(x=x_bar_np, P=P_bar_np)
        return matched_state

    @staticmethod
    def mixture_reduction(weights, states, threshold):
        """Uses a greedy merging method to reduce the number of Gaussian components
        for a Gaussian mixture density

        Args:
            weights ([type]): normalized weight of Gaussian components in logarithmic scale
            states ([type]): [description]
            threshold ([type]): [description]

        Returns:
            [type]: [description]
        """
        new_log_weights = []
        new_merged_states = []

        states = copy.deepcopy(states)
        weights = copy.deepcopy(weights)
        while states:
            # Find the component with the highest weight
            max_weight_idx = np.argmax(weights)
            try:
                S = np.linalg.inv(states[max_weight_idx].P)
            except np.linalg.LinAlgError:
                logging.error(f"Linalg error \n{states[max_weight_idx].P}")
                logging.warning("Get pseudo inverse matrix")
                S = np.linalg.pinv(states[max_weight_idx].P)

            idx_to_merge = []
            indices = list(range(len(states)))
            for idx in indices:
                state = states[idx]
                # Find Mahalanobis distance between state with max weight and current state
                x_diff = state.x - states[max_weight_idx].x
                mahalanobis_distance = x_diff.T @ S @ x_diff

                if mahalanobis_distance < threshold:
                    idx_to_merge.append(idx)

            # perform moment matching for states that close to state with max weights
            normalized_weights, log_sum_w = normalize_log_weights([weights[idx] for idx in idx_to_merge])
            merged_state = GaussianDensity.moment_matching(
                weights=normalized_weights, states=[states[idx] for idx in idx_to_merge]
            )

            # Remove merged states from original list of states
            for idx in reversed(idx_to_merge):
                del states[idx]
                del weights[idx]

            new_log_weights.append(log_sum_w)
            new_merged_states.append(merged_state)

        return new_log_weights, new_merged_states
