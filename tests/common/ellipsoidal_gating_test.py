import numpy as np
from scipy.stats import chi2

from mot.common.gaussian_density import GaussianDensity
from mot.common.state import Gaussian
from mot.configs import GroundTruthConfig, Object, SensorModelConfig
from mot.measurement_models import ConstantVelocityMeasurementModel
from mot.motion_models import ConstantVelocityMotionModel
from mot.simulator.measurement_data_generator import MeasurementData
from mot.simulator.object_data_generator import ObjectData


TOL = 1e-4


def test_ellipsoidal_gating():
    n_births = 1
    total_time = 10
    objects = [
        Object(
            initial=Gaussian(x=np.array([0.0, 0.0, 5.0, 5.0]), P=np.eye(4)),
            t_birth=0,
            t_death=10,
        )
    ]

    grount_truth_config = GroundTruthConfig(n_births=n_births, object_configs=objects, total_time=total_time)
    test_dt = 1.0
    test_sigma_q = 2.0
    motion_model = ConstantVelocityMotionModel(dt=test_dt, sigma_q=test_sigma_q)

    test_P_D = 1.0
    test_lambda_c = 60.0
    test_range_c = np.array([[-200.0, 200.0], [-200, 200]])
    sensor_model = SensorModelConfig(P_D=test_P_D, lambda_c=test_lambda_c, range_c=test_range_c)

    test_sigma_r = 10.0
    meas_model = ConstantVelocityMeasurementModel(sigma_r=test_sigma_r)

    object_data = ObjectData(grount_truth_config, motion_model, if_noisy=False)

    meas_data = MeasurementData(object_data=object_data, sensor_model=sensor_model, meas_model=meas_model)

    gating_size = chi2.ppf(0.99, df=meas_model.d)

    states = [None for i in range(grount_truth_config.total_time)]
    for timestep in range(0, grount_truth_config.total_time):
        states[timestep] = Gaussian(x=np.array(object_data[timestep].x), P=np.eye(motion_model.d))

        if timestep == 0:
            continue
        [z_ingate, meas_in_gate] = GaussianDensity.ellipsoidal_gating(
            state_prev=states[timestep - 1],
            z=np.array(meas_data[timestep]),
            measurement_model=meas_model,
            gating_size=gating_size,
        )
