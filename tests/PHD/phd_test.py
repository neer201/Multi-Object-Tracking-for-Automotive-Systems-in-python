from collections import namedtuple
from dataclasses import asdict

import numpy as np
import pytest

from mot.common.state import Gaussian, GaussianMixture, WeightedGaussian
from mot.configs import GroundTruthConfig, SensorModelConfig
from mot.measurement_models import ConstantVelocityMeasurementModel
from mot.motion_models import ConstantVelocityMotionModel
from mot.scenarios.scenario_configs import linear_full_mot
from mot.simulator import MeasurementData, ObjectData
from mot.trackers.multiple_object_trackers.PHD import GMPHD
from mot.utils.get_path import get_images_dir
from mot.utils.visualizer import Plotter


test_env_cases = [
    (
        linear_full_mot,
        ConstantVelocityMotionModel,
        ConstantVelocityMeasurementModel,
        "n MOT linear (CV)",
    ),
]


def generate_environment(config, motion_model, meas_model, *args, **kwargs):
    config = asdict(config)
    ground_truth = GroundTruthConfig(**config)
    motion_model = motion_model(**config)
    sensor_model = SensorModelConfig(**config)
    meas_model = meas_model(**config)
    object_data = ObjectData(ground_truth_config=ground_truth, motion_model=motion_model, if_noisy=False)
    meas_data = MeasurementData(object_data=object_data, sensor_model=sensor_model, meas_model=meas_model)
    env = namedtuple(
        "env",
        [
            "congig",
            "ground_truth",
            "motion_model",
            "sensor_model",
            "meas_model",
            "object_data",
            "meas_data",
        ],
    )
    return env(
        config,
        ground_truth,
        motion_model,
        sensor_model,
        meas_model,
        object_data,
        meas_data,
    )


@pytest.mark.parametrize("config,motion_model,meas_model,name", test_env_cases)
def test_generate_environment(config, motion_model, meas_model, name, *args, **kwargs):
    try:
        env = generate_environment(config, motion_model, meas_model)  # noqa F841
        pass
    except Exception:
        raise AssertionError


@pytest.mark.parametrize("config, motion_model, meas_model, name", test_env_cases)
def test_components_list_operations(config, motion_model, meas_model, name, *args, **kwargs):
    birth_model = GaussianMixture(
        [
            WeightedGaussian(log_weight=np.log(0.03), gaussian=Gaussian(x=pos, P=400 * np.eye(4)))
            for pos in [
                np.array([0, 0, 0, 0]),
                np.array([400, -600, 0, 0]),
                np.array([-800, -200, 0, 0]),
                np.array([-200, 800, 0, 0]),
            ]
        ]
    )
    gmm_components = GaussianMixture()
    gmm_components.extend(birth_model)


@pytest.mark.parametrize("config, motion_model, meas_model, name", test_env_cases)
def test_tracker_predict_step(config, motion_model, meas_model, name, *args, **kwargs):
    env = generate_environment(config, motion_model, meas_model)

    # Single object tracker parameter setting
    P_G = 0.99  # gating size in percentage
    w_minw = 1e-4  # hypothesis pruning threshold
    merging_threshold = 2  # hypothesis merging threshold
    M = 100  # maximum number of hypotheses kept in MHT

    birth_model = GaussianMixture(
        [
            WeightedGaussian(weight=np.log(0.03), gm=Gaussian(x=pos, P=400 * np.eye(4)))
            for pos in [
                np.array([0, 0, 0, 0]),
                np.array([400, -600, 0, 0]),
                np.array([-800, -200, 0, 0]),
                np.array([-200, 800, 0, 0]),
            ]
        ]
    )
    tracker = GMPHD(
        meas_model=env.meas_model,
        sensor_model=env.sensor_model,
        motion_model=env.motion_model,
        P_S=config.P_S,
        birth_model=birth_model,
        P_G=P_G,
        w_min=w_minw,
        merging_threshold=merging_threshold,
        M=M,
    )

    # One step predict
    tracker.predict_step()


@pytest.mark.parametrize("config, motion_model, meas_model, name", test_env_cases)
def test_tracker_predict_and_update_step(config, motion_model, meas_model, name, *args, **kwargs):
    env = generate_environment(config, motion_model, meas_model)

    # Single object tracker parameter setting
    P_G = 0.99  # gating size in percentage
    w_minw = 1e-3  # hypothesis pruning threshold
    merging_threshold = 2  # hypothesis merging threshold
    M = 100  # maximum number of hypotheses kept in MHT

    birth_model = GaussianMixture(
        [
            WeightedGaussian(log_weight=np.log(0.03), gaussian=Gaussian(x=pos, P=400 * np.eye(4)))
            for pos in [
                np.array([0, 0, 0, 0]),
                np.array([400, -600, 0, 0]),
                np.array([-800, -200, 0, 0]),
                np.array([-200, 800, 0, 0]),
            ]
        ]
    )
    tracker = GMPHD(
        meas_model=env.meas_model,
        sensor_model=env.sensor_model,
        motion_model=env.motion_model,
        P_S=config.P_S,
        birth_model=birth_model,
        P_G=P_G,
        P_D=0.6,
        w_min=w_minw,
        merging_threshold=merging_threshold,
        M=M,
    )

    # One step predict
    tracker.predict_step(dt=1.0)

    test_measurements = np.array([config.object_configs[idx].initial_state.x[0:2] for idx in range(3)])
    # One step update
    tracker.update(z=test_measurements)


@pytest.mark.parametrize("config, motion_model, meas_model, name", test_env_cases)
def test_tracker_estimate(config, motion_model, meas_model, name, *args, **kwargs):
    env = generate_environment(config, motion_model, meas_model)

    # Single object tracker parameter setting
    P_G = 0.99  # gating size in percentage
    w_minw = 1e-4  # hypothesis pruning threshold
    merging_threshold = 2  # hypothesis merging threshold
    M = 100  # maximum number of hypotheses kept in MHT
    P_D = 0.9

    birth_model = GaussianMixture(
        [
            WeightedGaussian(log_weight=np.log(0.03), gaussian=Gaussian(x=pos, P=400 * np.eye(4)))
            for pos in [
                np.array([0, 0, 0, 0]),
                np.array([400, -600, 0, 0]),
                np.array([-800, -200, 0, 0]),
                np.array([-200, 800, 0, 0]),
            ]
        ]
    )
    tracker = GMPHD(
        meas_model=env.meas_model,
        sensor_model=env.sensor_model,
        motion_model=env.motion_model,
        P_S=config.P_S,
        birth_model=birth_model,
        P_G=P_G,
        P_D=P_D,
        w_min=w_minw,
        merging_threshold=merging_threshold,
        M=M,
    )

    # One step predict
    tracker.predict_step(dt=1.0)

    test_measurements = np.array([config.object_configs[idx].initial_state.x[0:2] for idx in range(3)])
    # One step update
    tracker.update(z=test_measurements)

    # One step estimate
    estimates = tracker.PHD_estimator()  # noqa F841


@pytest.mark.parametrize("config, motion_model, meas_model, name", test_env_cases)
def test_tracker_normal(config, motion_model, meas_model, name, *args, **kwargs):

    birth_model = GaussianMixture(
        [
            WeightedGaussian(log_weight=np.log(0.03), gaussian=Gaussian(x=pos, P=400 * np.eye(4)))
            for pos in [
                np.array([0, 0, 0, 0]),
                np.array([400, -600, 0, 0]),
                np.array([-800, -200, 0, 0]),
                np.array([-200, 800, 0, 0]),
            ]
        ]
    )
    env = generate_environment(config, motion_model, meas_model)

    # Single object tracker parameter setting
    P_G = 0.9  # gating size in percentage
    w_minw = 1e-4  # hypothesis pruning threshold
    merging_threshold = 4  # hypothesis merging threshold
    M = 100  # maximum number of hypotheses kept in MHT
    P_D = 0.99

    tracker = GMPHD(
        meas_model=env.meas_model,
        sensor_model=env.sensor_model,
        motion_model=env.motion_model,
        P_S=config.P_S,
        birth_model=birth_model,
        P_G=P_G,
        P_D=P_D,
        w_min=w_minw,
        merging_threshold=merging_threshold,
        M=M,
    )

    estimations = tracker.estimate(env.meas_data)

    # Animator.animate(
    #     [env.meas_data, env.object_data, list(estimations)],
    #     title=name,
    #     filename=get_images_dir(__file__) + "/" + "meas_data_and_obj_data" + ".gif",
    # )

    Plotter.plot_several(
        [env.meas_data, env.object_data, list(estimations)],
        out_path=get_images_dir(__file__) + "/" + "meas_data_and_obj_data_and_estimations" + ".png",
    )

    Plotter.plot_several(
        [list(estimations)],
        out_path=get_images_dir(__file__) + "/" + "estimations" + ".png",
    )
