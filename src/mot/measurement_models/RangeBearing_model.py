import numpy as np

from .base_measurement_model import MeasurementModel


class RangeBearingMeasurementModel(MeasurementModel):
    def __init__(self, sigma_r: float, sigma_b: float, sensor_pos, *args, **kwargs):
        """Creats the range/bearnig measurement model

        Parameters
        ----------
        sigma_r : scalar
            standart deviation of measurement noise added to range
        sigma_b : scalar
            standart deviation of measurement noise added to bearing
        sensor_pos : np.array (2 x 1)
            sensor positions

        Attributes
        -------
        sigma_r : scalar
            standart deviation of measurement noise added to range
        # TODO beatify
        d (scalar): measurement dimenstion
            H (2 x 4 matrix): function handle return an observation matrix
            R (2 x 2 matrix): measurement noise covatiance
            h (2 x 1 matrix): function handle return a measurement


        Returns
        -------
        [type]
            [description]
        """

        self.d = 2
        self.sigma_r = sigma_r
        self.sigma_b = sigma_b
        self.R = np.diag([np.power(self.sigma_r, 2), np.power(self.sigma_b, 2)])
        self.sensor_pos = sensor_pos
        super(RangeBearingMeasurementModel, self).__init__(*args, **kwargs)

    def __call__(self, x):
        return self.H @ x

    @property
    def _observation_matrix(self):
        return self.H()

    @property
    def _noise_covariance(self):
        return self.R

    def observe(self, state_vector):
        observation = np.array([self._get_range(state_vector), self._get_bearing(state_vector)])
        return observation

    def _get_range(self, state_vector):
        assert isinstance(state_vector, np.ndarray)
        return np.linalg.norm(state_vector[:2] - self.sensor_pos)

    def _get_bearing(self, state_vector):
        return np.arctan2(state_vector[1] - self.sensor_pos[1], state_vector[0] - self.sensor_pos[0])

    def H(self, state_vector=None):
        # yapf: disable
        x, s, rng = state_vector, self.sensor_pos, self._get_range(state_vector)
        H = np.array([
            [(x[0] - s[0]) / rng, (x[1] - s[1]) / rng, 0.0, 0.0, 0.0],
            [-(x[1] - s[1]) / rng**2, (x[0] - s[0]) / rng**2, 0.0, 0.0, 0.0],
        ])
        # yapf: enable
        return H

    @property
    def dim(self):
        return self.d

    def h(self, state_vector: np.ndarray) -> np.ndarray:
        """Handle to generate measurement

        Parameters
        ----------
        state_vector : np.ndarray
            state vector

        Returns
        -------
        np.ndarray
            state_vector
        """
        return np.array([self._get_range(state_vector), self._get_bearing(state_vector)])
