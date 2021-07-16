from typing import List

import numpy as np
from scipy.stats import chi2
from tqdm import tqdm as tqdm

from mot.common.gaussian_density import GaussianDensity
from mot.common.state import Gaussian
from mot.configs import SensorModelConfig
from mot.measurement_models import MeasurementModel
from mot.motion_models import MotionModel

from .base_n_object_tracker import KnownObjectTracker


class GlobalNearestNeighboursTracker(KnownObjectTracker):
    def __init__(
        self,
        meas_model: MeasurementModel,
        sensor_model: SensorModelConfig,
        motion_model: MotionModel,
        M,
        merging_threshold,
        P_G,
        w_min,
        *args,
        **kwargs,
    ) -> None:
        self.meas_model = meas_model
        self.sensor_model = sensor_model
        self.motion_model = motion_model
        self.w_min = w_min
        self.P_G = P_G
        self.gating_size = chi2.ppf(P_G, df=self.meas_model.d)
        self.M = M
        self.n = 5
        self.merging_threshold = merging_threshold
        self.hypotheses_weight = None
        self.multi_hypotheses_bank = None
        super(GlobalNearestNeighboursTracker).__init__()

    @property
    def method(self) -> str:
        return "GNN"

    def estimate(self, initial_states: List[Gaussian], measurements, verbose=False):
        """Tracks a single object using Gauss sum filtering

        For each filter recursion iteration implemented next steps:
        1) for each prior perform prediction
        2) for each hypothesis, perform ellipsoidal gating
           and only create object detection hypotheses for detections
        3) construct 2D cost matrix of size
           (number of objects x number of z_ingate + number of objects)
        4) find best assignment using a 2D assignment solver
        5) create new local hypotheses accotding to the best assgnment matrix obtained
        6) get obhect state estimates
        7) preform prefict for each local hypotheses
        """

        previous_states = initial_states
        estimations = [None for x in range(len(measurements))]

        for timestep, measurements_in_scene in tqdm(enumerate(measurements)):
            estimations[timestep] = self.estimation_step(
                object_states=previous_states,
                current_measurements=np.array(measurements_in_scene),
            )
            previous_states = [
                GaussianDensity.predict(state, self.motion_model, dt=1.0) for state in estimations[timestep]
            ]
        return tuple(estimations)

    def estimation_step(self, object_states: List[Gaussian], current_measurements: np.ndarray):
        # 1) elipsoidal gating separately for each object
        num_of_measurements = current_measurements.shape[0]
        indices_of_objects_in_gate = np.zeros((self.n, num_of_measurements), dtype=bool)

        for object_state_idx, object_state in enumerate(object_states):
            z_ingate, indices_in_gate = GaussianDensity.ellipsoidal_gating(
                object_state,
                current_measurements,
                measurement_model=self.meas_model,
                gating_size=self.gating_size,
            )
            indices_of_objects_in_gate[object_state_idx, :] = indices_in_gate

        # Disconsider measurements which do not fall inside any object gate
        mask_to_keep = np.sum(indices_of_objects_in_gate, axis=0, dtype=bool)
        indices_to_keep = np.where(mask_to_keep)[0]  # TODO fix it
        filtered_measurements = current_measurements[indices_to_keep]
        num_of_filtered_measurements = filtered_measurements.shape[0]

        #   3) construct 2D cost matrix of size
        #    (number of objects x number of z_ingate + number of objects)
        cost_matrix = np.full((self.n, num_of_filtered_measurements + self.n), np.inf)
        w_theta_factor = np.log(self.sensor_model.P_D / self.sensor_model.intensity_c)
        w_theta_0 = np.log(1 - self.sensor_model.P_D)  # misdetection
        for idx_object in range(self.n):
            for enum_meas, idx_meas in enumerate(indices_to_keep):
                S_i_h = (
                    self.meas_model.H(object_states[idx_object])
                    @ object_states[idx_object].P
                    @ self.meas_model.H(object_states[idx_object]).T
                )
                z_bar_i_h = self.meas_model.h(object_states[idx_object].x)
                vec_diff = current_measurements[idx_meas] - z_bar_i_h
                mahl = 0.5 * vec_diff @ np.linalg.inv(S_i_h) @ vec_diff.T
                factor = 0.5 * np.log(np.linalg.det(2 * np.pi * S_i_h))
                cost = mahl + factor - w_theta_factor

                cost_matrix[idx_object, enum_meas] = cost
            cost_matrix[idx_object, num_of_filtered_measurements + idx_object] = w_theta_0

        # 4) find best assignment using a 2D assignment solver

        # try:
        #     indices, assignment = linear_sum_assignment(cost_matrix=cost_matrix)
        # except ValueError as e:
        #     print(e)
        # print(assignment)
        # import pdb; pdb.set_trace()
        # for idx_object in range(self.n): # row in cost matrix
        #     if assignment[idx_object] <=
        estimates = object_states
        return estimates
