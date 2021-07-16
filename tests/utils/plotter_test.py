from collections import namedtuple
from dataclasses import asdict

import numpy as np
import pytest

from mot.common.state import Gaussian
from mot.configs import GroundTruthConfig, SensorModelConfig
from mot.measurement_models import ConstantVelocityMeasurementModel
from mot.motion_models import ConstantVelocityMotionModel
from mot.scenarios.scenario_configs import linear_full_mot
from mot.simulator import MeasurementData, ObjectData
from mot.utils.get_path import get_images_dir
from mot.utils.visualizer import Animator, Plotter


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
    estimations = [
        [
            Gaussian(x=pos, P=400 * np.eye(4))
            for pos in [
                np.array([0, 0, 0, 0]),
                np.array([400, -600, 0, 0]),
                np.array([-800, -200, 0, 0]),
                np.array([-200, 800, 0, 0]),
            ]
        ]
    ] * 10
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
            "estimations",
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
        estimations,
    )


@pytest.mark.parametrize("config, motion_model, meas_model, name", test_env_cases)
def test_plot_object_data(config, motion_model, meas_model, name, *args, **kwargs):
    env = generate_environment(config, motion_model, meas_model)
    Plotter.plot(
        env.object_data,
        out_path=get_images_dir(__file__) + "/" + "obj_data" + ".png",
    )


@pytest.mark.parametrize("config, motion_model, meas_model, name", test_env_cases)
def test_plot_meas_data(config, motion_model, meas_model, name, *args, **kwargs):
    env = generate_environment(config, motion_model, meas_model)
    Plotter.plot(
        env.meas_data,
        out_path=get_images_dir(__file__) + "/" + "meas_data" + ".png",
    )


@pytest.mark.parametrize("config, motion_model, meas_model, name", test_env_cases)
def test_plot_object_meas_data(config, motion_model, meas_model, name, *args, **kwargs):
    env = generate_environment(config, motion_model, meas_model)
    Plotter.plot_several(
        [env.meas_data, env.object_data],
        out_path=get_images_dir(__file__) + "/" + "meas_data_and_obj_data" + ".png",
    )


def test_plot_one_gaussian(*args, **kwargs):
    gaussian = Gaussian(x=np.array([0, 0, 10, 10]), P=np.diag([400, 200, 0, 0]))

    Plotter.plot(
        [gaussian],
        lim_x=(-200, 200),
        lim_y=(-200, 200),
        out_path=get_images_dir(__file__) + "/" + "one_gaussian" + ".png",
    )


@pytest.mark.parametrize("config, motion_model, meas_model, name", test_env_cases)
def test_plot_gaussians(config, motion_model, meas_model, name, *args, **kwargs):
    env = generate_environment(config, motion_model, meas_model)
    Plotter.plot_several(
        [env.estimations],
        out_path=get_images_dir(__file__) + "/" + "gaussians" + ".png",
    )


@pytest.mark.parametrize("config, motion_model, meas_model, name", test_env_cases)
def test_animate_input_data(config, motion_model, meas_model, name, *args, **kwargs):
    env = generate_environment(config, motion_model, meas_model)
    Animator.animate(
        [env.meas_data, env.object_data],
        title=name,
        filename=get_images_dir(__file__) + "/" + "meas_data_and_obj_data" + ".gif",
    )
