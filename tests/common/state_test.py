import unittest

import numpy as np
import pytest

from mot.common.state import Gaussian


TOL = 1e-4


class Test_Gaussian(unittest.TestCase):
    def test_create_state(self):
        state = Gaussian(x=np.array([0, 0]), P=np.eye(2))  # noqa F841
        assert True

    def test_wrong_x_type(self):
        with pytest.raises(Exception) as e_info:
            state = Gaussian(x=0, P=np.eye(2))  # noqa F841
        assert str(e_info.value) == "Argument of wrong type!"

    def test_wrong_P_type(self):
        with pytest.raises(Exception) as e_info:
            state = Gaussian(x=np.array([0]), P=0)  # noqa F841
        assert str(e_info.value) == "Argument of wrong type!"

    def test_wrong_P_dim(self):
        with pytest.raises(Exception) as e_info:
            state = Gaussian(x=np.array([0, 0]), P=np.array([[0, 1], [0, 1], [0, 1]]))  # noqa F841
        assert str(e_info.value) == "Covariance matrix should be square!"

    def test_diffrenet_dims(self):
        with pytest.raises(Exception) as e_info:
            state = Gaussian(x=np.array([0, 0, 0]), P=np.eye(2))  # noqa F841
        assert str(e_info.value) == "size of vector should be equal P column size!"
