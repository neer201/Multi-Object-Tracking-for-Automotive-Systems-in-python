from abc import ABC

import numpy as np


class MeasurementModel(ABC):
    """
    MeasurementModel is a abstract class for different measurement models.
    """

    def __init__(self, random_state=None, *args, **kwargs):
        self._generator = np.random.RandomState(random_state)

    def __repr__(self):
        pass

    def observe(self, params):
        pass

    @property
    def dim(self):
        pass
