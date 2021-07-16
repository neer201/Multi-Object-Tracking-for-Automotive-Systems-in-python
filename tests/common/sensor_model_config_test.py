import unittest

import numpy as np

from mot.configs import SensorModelConfig


TOL = 1e-4


class Test_SensorModelConfig(unittest.TestCase):
    def test_sensor_model_config_2D(self):
        test_P_D = 0.67877834167146323
        test_lambda_c = 0.21275901010320308
        test_range_c = np.array(
            [
                [-0.081623375045665081, 0.27447940788559888],
                [-0.8675183864664644, 0.55935706739745583],
            ]
        )

        expected_pdf_c = 1.96806104572303
        expected_intensity_c = 0.41872271991070659

        sensor_model = SensorModelConfig(P_D=test_P_D, lambda_c=test_lambda_c, range_c=test_range_c)

        assert abs(sensor_model.pdf_c - expected_pdf_c) < TOL, (
            f"The value of clutter pdf 2D case: " f"Expected {expected_pdf_c} " f"Got {sensor_model.pdf_c}"
        )

        assert abs(sensor_model.intensity_c - expected_intensity_c) < TOL, (
            f"Clutter intensity 2D case: " f"Expected {expected_intensity_c}, " f"got {sensor_model.intensity_c}"
        )

    def test_sensor_model_config_1D(self):
        test_P_D = 0.8147
        test_lambda_c = 0.9058
        test_range_c = np.array([-0.1270, 0.9134])

        expected_pdf_c = 0.961203267377758
        expected_intensity_c = 0.870650169481514

        sensor_model = SensorModelConfig(P_D=test_P_D, lambda_c=test_lambda_c, range_c=test_range_c)

        assert abs(sensor_model.pdf_c - expected_pdf_c) < TOL, (
            f"The value of clutter pdf 2D case: " f"Expected {expected_pdf_c} " f"Got {sensor_model.pdf_c}"
        )
        assert abs(sensor_model.intensity_c - expected_intensity_c) < TOL, (
            f"Clutter intensity 2D case: " f"Expected {expected_intensity_c}, " f"Got {sensor_model.intensity_c}"
        )
