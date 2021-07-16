from dataclasses import dataclass

import numpy as np


@dataclass
class Estimation:
    track_id: int
    state_estimation: np.ndarray
