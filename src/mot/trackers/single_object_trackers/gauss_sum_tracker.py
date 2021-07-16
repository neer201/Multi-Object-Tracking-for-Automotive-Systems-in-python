import numpy as np
from scipy.stats import chi2
from tqdm import tqdm as tqdm

from ...common import (
    Gaussian,
    GaussianDensity,
    HypothesisReduction,
    normalize_log_weights,
)
from ...configs import SensorModelConfig
from ...measurement_models import MeasurementModel
from ...motion_models import MotionModel
from .base_single_object_tracker import SingleObjectTracker


class GaussSumTracker(SingleObjectTracker):
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
        self.merging_threshold = merging_threshold
        self.hypotheses_weight = None
        self.multi_hypotheses_bank = None

        super(GaussSumTracker).__init__()

    def estimate(self, initial_state: Gaussian, measurements, verbose=False):
        """Tracks a single object using Gauss sum filtering

        For each filter recursion iteration implemented next steps:
        1) for each hypothesis, create missed detection hypothesis
        2) for each hypothesis, perform ellipsoidal gating
           and only create object detection hypotheses for detections
           inside the gate
        3) normalise hypotheses weights
        4) prune hypotheses with small weights and then re-normalise the weights
        5) hypothese merging
        6) cap the number of the hypotheses and then re-normalise the weights
        7) extract object state estimate using the most probably
           hypothesis estimation
        8) for each hypothesis, perform prediction
        """

        prev_state = initial_state
        estimations = [None for x in range(len(measurements))]
        self.hypotheses_weight = [np.log(1.0)]
        self.multi_hypotheses_bank = [initial_state]

        for timestep, measurements_in_scene in tqdm(enumerate(measurements)):
            estimations[timestep] = self.estimation_step(
                predicted_state=prev_state,
                current_measurements=np.array(measurements_in_scene),
            )
            prev_state = GaussianDensity.predict(state=estimations[timestep], motion_model=self.motion_model)
        return tuple(estimations)

    def estimation_step(self, predicted_state: Gaussian, current_measurements: np.ndarray):
        new_hypotheses, new_weights = [], []
        w_theta_factor = np.log(self.sensor_model.P_D / self.sensor_model.intensity_c)
        w_theta_0 = np.log(1 - self.sensor_model.P_D)  # misdetection

        for _old_idx, (curr_weight, curr_hypothesis) in enumerate(
            zip(self.hypotheses_weight, self.multi_hypotheses_bank)
        ):
            # 1) for each hypothesis, create missed detection hypothesis
            new_hypotheses.append(curr_hypothesis)
            new_weights.append(w_theta_0 + curr_weight)

            # 2) for each hypothesis, perform ellipsoidal gating
            # and only create object detection hypotheses for detection
            # inside the gate
            z_ingate, _ = GaussianDensity.ellipsoidal_gating(
                curr_hypothesis,
                current_measurements,
                self.meas_model,
                self.gating_size,
            )

            predicted_likelihood = GaussianDensity.predicted_likelihood(curr_hypothesis, z_ingate, self.meas_model)

            # for each measurement create detection hypotheses
            for idx, meausurement in z_ingate:
                new_hypotheses.append(GaussianDensity.update(curr_hypothesis, meausurement, self.meas_model))
                new_weights.append(predicted_likelihood[idx] + w_theta_factor)

        self.hypotheses_weight.extend(new_weights)
        self.multi_hypotheses_bank.extend(new_hypotheses)
        assert len(self.hypotheses_weight) == len(self.multi_hypotheses_bank)

        # 3.normalise hypotheses weights
        self.hypotheses_weight, _ = normalize_log_weights(self.hypotheses_weight)

        # 4. Prune hypotheses with small weights and then re-normalise the weights
        self.hypotheses_weight, self.multi_hypotheses_bank = HypothesisReduction.prune(
            self.hypotheses_weight, self.multi_hypotheses_bank, threshold=self.w_min
        )
        self.hypotheses_weight, _ = normalize_log_weights(self.hypotheses_weight)

        # 5. Hypotheses merging and normalize
        self.hypotheses_weight, self.multi_hypotheses_bank = HypothesisReduction.merge(
            self.hypotheses_weight,
            self.multi_hypotheses_bank,
            threshold=self.merging_threshold,
        )
        self.hypotheses_weight, _ = normalize_log_weights(self.hypotheses_weight)

        # 6. Cap the number of the hypotheses and then re-normalise the weights
        self.hypotheses_weight, self.multi_hypotheses_bank = HypothesisReduction.cap(
            self.hypotheses_weight, self.multi_hypotheses_bank, top_k=self.M
        )
        self.hypotheses_weight, _ = normalize_log_weights(self.hypotheses_weight)

        # 7. Get object state from the most probable hypothesis
        if self.multi_hypotheses_bank:
            current_step_state = self.multi_hypotheses_bank[np.argmax(self.hypotheses_weight)]
            estimation = current_step_state
        else:
            estimation = predicted_state

        # 8. For each hypotheses do prediction
        self.updated_states = [
            GaussianDensity.predict(hypothesis, self.motion_model) for hypothesis in self.multi_hypotheses_bank
        ]
        self.multi_hypotheses_bank = self.updated_states
        return estimation

    @property
    def method(self):
        return "gauss sum filter"
