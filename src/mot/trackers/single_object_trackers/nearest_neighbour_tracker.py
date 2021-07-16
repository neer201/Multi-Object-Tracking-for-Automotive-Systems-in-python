import numpy as np

from mot.common.gaussian_density import GaussianDensity
from mot.common.state import Gaussian
from mot.configs import SensorModelConfig
from mot.measurement_models import MeasurementModel
from mot.motion_models import MotionModel
from .base_single_object_tracker import SingleObjectTracker


class NearestNeighbourTracker(SingleObjectTracker):
    def __init__(
        self,
        gating_size: float,
        meas_model: MeasurementModel,
        sensor_model: SensorModelConfig,
        motion_model: MotionModel,
        *args,
        **kwargs,
    ) -> None:
        self.meas_model = meas_model
        self.sensor_model = sensor_model
        self.motion_model = motion_model
        self.gating_size = gating_size
        super().__init__()

    @property
    def name(self):
        return "Nearest Neighbout SOT"

    def estimate(self, initial_state: Gaussian, measurements):
        """Tracks a single object using nearest neighbour association

        For each filter recursion iteration implemented next steps:
        1) gating
        2) calculates the predicted likelihood for each measurement in the gate
        3) find the nearest neighbour measurement
        4) compares the weight of the missed detection hypotheses and
           the weight of the object detection hypothesis created using
           the nearest neigbour measurement
        5) if the object detection hypothesis using the nearest neighbour
           measurement has the hightes weight, perform Kalman update
        6) extract object state estimate
        7) prediction
        """
        prev_state = initial_state
        estimations = [None for x in range(len(measurements))]
        for timestep, measurements_in_scene in enumerate(measurements):
            estimations[timestep] = self.estimation_step(
                predicted_state=prev_state,
                current_measurements=np.array(measurements_in_scene),
            )
            prev_state = GaussianDensity.predict(state=estimations[timestep], motion_model=self.motion_model, dt=1.0)
        return tuple(estimations)

    def estimation_step(self, predicted_state: Gaussian, current_measurements: np.ndarray):
        # 1. Gating

        (meas_in_gate, _) = GaussianDensity.ellipsoidal_gating(
            state_prev=predicted_state,
            z=current_measurements,
            measurement_model=self.meas_model,
            gating_size=self.gating_size,
        )
        if meas_in_gate.size == 0:  # number of hypothesis
            current_step_state = predicted_state

        else:
            # 2. Calculate the predicted likelihood for each measurement in the gate

            predicted_likelihood = GaussianDensity.predict_loglikelihood(
                state_pred=predicted_state,
                z=meas_in_gate,
                measurement_model=self.meas_model,
            )

            # Hypothesis evaluation
            # detection
            w_theta_factor = np.log(self.sensor_model.P_D / self.sensor_model.intensity_c)
            w_theta_k = predicted_likelihood + w_theta_factor
            # misdetection
            w_theta_0 = 1 - self.sensor_model.P_D

            # 3. Compare the weight of the missed detection
            # hypothesis and the weight of the object detection hypothesis
            # using the nearest neighbour measurement
            max_k = np.argmax(w_theta_k)
            max_w_theta_k = w_theta_k[max_k]

            if w_theta_0 < max_w_theta_k:
                # nearest neighbour measurement
                z_NN = meas_in_gate[max_k]
                z_NN = np.atleast_2d(z_NN)
                current_step_state = GaussianDensity.update(
                    state_pred=predicted_state,
                    z=z_NN,
                    measurement_model=self.meas_model,
                )
            else:
                current_step_state = predicted_state
        estimation = current_step_state
        return estimation
