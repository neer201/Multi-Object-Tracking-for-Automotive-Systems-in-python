import unittest

from mot.measurement_models import ConstantVelocityMeasurementModel


class Test_MotionModels(unittest.TestCase):
    def test_constant_velocity_motion_model(self):
        sigma_r = 10.0
        meas_model = ConstantVelocityMeasurementModel(sigma_r=sigma_r)  # noqa F841
        raise NotImplementedError
        # TODO: write test

        # initial_state = np.array([0.0, 0.0, 5.0, 5.0])
        # expected_predicted_array = np.array([5.0, 5.0, 5.0, 5.0])
        # predicted_state = CV_motion_model.move(
        #     state=Gaussian(x=initial_state, P=np.eye(4))
        # )

        # np.testing.assert_array_equal(
        #     predicted_state.x,
        #     expected_predicted_array,
        #     err_msg=f"Wrong prediction state! {predicted_state}",
        # )
