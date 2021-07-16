import unittest

import numpy as np

from mot.common.normalize_log_weights import normalize_log_weights


class Test_normalize_log_weights(unittest.TestCase):
    def test_normalize_log_weights_single_element(self):
        test_weights = [10.0]
        expected_log_sum = 0.0
        # expected_log_norm_weights = [0.0]
        eps = 1e-4

        actual_log_norm_weights, actual_log_sum = normalize_log_weights(test_weights)

        assert np.abs(actual_log_sum - expected_log_sum) > eps, (
            f"check log sum calculation: " f"actual_log_sum = {actual_log_sum}"
        )
