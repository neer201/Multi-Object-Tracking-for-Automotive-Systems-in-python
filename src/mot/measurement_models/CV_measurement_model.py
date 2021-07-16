import numpy as np

from .base_measurement_model import MeasurementModel


class ConstantVelocityMeasurementModel(MeasurementModel):
    def __init__(self, sigma_r, *args, **kwargs):
        """Creates the measurement model for a 2D nearly constant velocity motion model

        Args:
            sigma (scalar): standart deviation of a measurement noise

        Attributes:
            d (scalar): measurement dimenstion
            H (2 x 4 matrix): function handle return an observation matrix
            R (2 x 2 matrix): measurement noise covatiance
            h (2 x 1 matrix): function handle return a measurement

        Returns:
            self (MeasurementModel)
        """

        self.d = 2
        self.R = (sigma_r ** 2) * np.eye(2)
        super(ConstantVelocityMeasurementModel, self).__init__(*args, **kwargs)

    def __call__(self, x):
        return self.H @ x

    def __repr__(self) -> str:
        return self.__class__.__name__ + (f"(d={self.d}, " f"R={self.R}, ")

    def _observation_matrix(self, state_vector):
        assert isinstance(state_vector, np.ndarray)
        return self.H(state_vector)

    @property
    def _noise_covariance(self):
        return self.R

    def observe(self, state_vector):
        assert isinstance(state_vector, np.ndarray)
        Z = self._generator.multivariate_normal(
            mean=self._observation_matrix(state_vector) @ state_vector,
            cov=self._noise_covariance,
        )
        return Z

    def H(self, state_vector):
        return np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])

    def h(self, state_vector):
        return self.H(state_vector) @ state_vector

    @property
    def dim(self):
        return self.d


class NuscenesConstantVelocityMeasurementModel(MeasurementModel):
    def __init__(self, sigma_r, *args, **kwargs):
        """Creates the measurement model for a 2D nearly constant velocity motion model

        Args:
            sigma (scalar): standart deviation of a measurement noise

        Attributes:
            d (scalar): measurement dimenstion
            H (2 x 4 matrix): function handle return an observation matrix
            R (2 x 2 matrix): measurement noise covatiance
            h (2 x 1 matrix): function handle return a measurement

        Returns:
            self (MeasurementModel)
        """

        self.d = 3
        self.R = (sigma_r ** 2) * np.eye(3)
        super(ConstantVelocityMeasurementModel, self).__init__(*args, **kwargs)

    def __call__(self, x):
        return self.H @ x

    def __repr__(self) -> str:
        return self.__class__.__name__ + (f"(d={self.d}, " f"R={self.R}, ")

    def _observation_matrix(self, state_vector):
        assert isinstance(state_vector, np.ndarray)
        return self.H(state_vector)

    @property
    def _noise_covariance(self):
        return self.R

    def observe(self, state_vector):
        assert isinstance(state_vector, np.ndarray)
        Z = self._generator.multivariate_normal(
            mean=self._observation_matrix(state_vector) @ state_vector,
            cov=self._noise_covariance,
        )
        return Z

    def H(self, state_vector):
        return np.concatenate((np.eye(3), np.zeros((3, 3))), axis=1)

    def h(self, state_vector):
        return self.H(state_vector) @ state_vector

    @property
    def dim(self):
        return self.d
