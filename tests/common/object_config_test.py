import unittest

import numpy as np
import pytest

from mot.common.state import Gaussian
from mot.configs import Object


TOL = 1e-4


class Test_ObjectConfig(unittest.TestCase):
    def test_create_object_config(self):
        state = Gaussian(x=np.array([0, 0]), P=np.eye(2))
        object_config = Object(initial=state, t_birth=0, t_death=10)  # noqa F841

    def test_birth_after_death(self):
        state = Gaussian(x=np.array([0, 0]), P=np.eye(2))
        with pytest.raises(Exception):
            object_config = Object(initial=state, t_birth=10, t_death=0)  # noqa F841
