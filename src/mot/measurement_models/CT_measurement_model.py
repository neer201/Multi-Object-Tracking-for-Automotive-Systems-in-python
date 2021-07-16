import numpy as np

from .base_measurement_model import MeasurementModel


class CoordinateTurnMeasurementModel(MeasurementModel):
    def __init__(self, sigma, *args, **kwargs):
        """Creates the measurement model for a 2D coordinate turn motion model

        Args:
            sigma (scalar): standart deviation of measurement noise

        Attributes:
            d (scalar): measurement dimenstion
            H (2 x 5 matrix): function handle return an observation matrix
            R (2 x 2 matrix): measurement noise covatiance
            h (2 x 1 matrix): function handle return a measurement

        Notes: the first two entries of the state vector represents
               the X-position and Y-position, respectively.

        Returns:
            self (MeasurementModel)
        """

        self.d = 2
        self.H = np.array([[1.0, 0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0, 0.0]])
        self.R = (sigma ** 2) * np.eye(2)

    def __call__(self, x):
        return self.H @ x

    @property
    def dim(self):
        return self.d
