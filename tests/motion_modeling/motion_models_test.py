import unittest

import numpy as np

from mot.common.state import Gaussian
from mot.motion_models import ConstantVelocityMotionModel


class Test_MotionModels(unittest.TestCase):
    def test_constant_velocity_motion_model(self):
        delta_t = 1.0
        sigma_q = 2.0
        CV_motion_model = ConstantVelocityMotionModel(dt=delta_t, sigma_q=sigma_q)

        initial_state = np.array([0.0, 0.0, 5.0, 5.0])
        expected_predicted_array = np.array([5.0, 5.0, 5.0, 5.0])
        predicted_state = CV_motion_model.move(state=Gaussian(x=initial_state, P=np.eye(4)))

        np.testing.assert_array_equal(
            predicted_state.x,
            expected_predicted_array,
            err_msg=f"Wrong prediction state! {predicted_state}",
        )
