import numpy as np


class SensorModelConfig:
    def __init__(self, P_D: float, lambda_c: float, range_c: np.ndarray, *args, **kwargs):
        """Creates the sensor model

        Parameters
        ----------
        P_D : float
            object detection probability
        lambda_c : float
            average number of clutter measurements per time scan, Poisson distributes
        range_c : np.ndarray
            range of surveillance area
            if 2D model: 2 x 2 matrix of the form [[xmin xmax], [ymin ymax]]
            if 1D model: 1 x 2 vector of the form [xmin xmax]

        Attributes
        ----------
        pdf_c : float
            uniform clutter density
        intensity_c : float
            uniform clutter intensity
        """
        self.P_D = P_D
        self.lambda_c = lambda_c
        self.range_c = range_c
        self.V = np.prod(np.diff(self.range_c))  # Volume
        self.pdf_c = 1 / self.V  # Spatial PDF
        self.intensity_c = self.lambda_c / self.V  # expected number of clutter detections per unit volume

    def __repr__(self) -> str:
        return self.__class__.__name__ + (
            f"(P_D={self.P_D}, "
            f"lambda_c={self.lambda_c}, "
            f"range_c={self.range_c}, "
            f"V={self.V}, "
            f"pdf_c = {self.pdf_c} "
            f"intensity_c = {self.intensity_c}"
        )
