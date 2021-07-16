import os
from dataclasses import asdict

import pytest

from mot.configs import GroundTruthConfig, SensorModelConfig
from mot.measurement_models import (
    ConstantVelocityMeasurementModel,
    RangeBearingMeasurementModel,
)
from mot.motion_models import ConstantVelocityMotionModel, CoordinateTurnMotionModel
from mot.scenarios.scenario_configs import (
    linear_n_mot,
    linear_sot,
    nonlinear_n_mot,
    nonlinear_sot,
)
from mot.simulator import MeasurementData
from mot.simulator.object_data_generator import ObjectData
from mot.utils.visualizer import Animator, Plotter


dir_path = os.path.dirname(os.path.realpath(__file__))

test_data = [
    (
        linear_sot,
        ConstantVelocityMotionModel,
        "linear_sot_CV.png",
        ConstantVelocityMeasurementModel,
    ),
    (
        nonlinear_sot,
        CoordinateTurnMotionModel,
        "nonlinear_sot_CT.png",
        RangeBearingMeasurementModel,
    ),
    (
        linear_n_mot,
        ConstantVelocityMotionModel,
        "linear_n_mot_CV.png",
        ConstantVelocityMeasurementModel,
    ),
    (
        nonlinear_n_mot,
        CoordinateTurnMotionModel,
        "nonlinear_n_mot_CT.png",
        RangeBearingMeasurementModel,
    ),
]


@pytest.mark.parametrize("config, motion_model, output_image_name, meas_model", test_data)
def test_linear_model_without_noise(config, motion_model, output_image_name, meas_model):
    config = asdict(config)
    ground_truth = GroundTruthConfig(**config)
    motion_model = motion_model(**config)

    object_data = ObjectData(ground_truth_config=ground_truth, motion_model=motion_model, if_noisy=False)

    sensor_model = SensorModelConfig(**config)

    meas_model = meas_model(**config)
    meas_data = MeasurementData(object_data=object_data, sensor_model=sensor_model, meas_model=meas_model)

    Plotter.plot([meas_data[0]], title=output_image_name, out_path="meas_" + output_image_name)

    Plotter.plot(
        [meas_data[0], object_data],
        title=output_image_name,
        out_path="meas_" + output_image_name,
    )

    Animator.animate(
        [object_data, meas_data],
        title=output_image_name,
        filename=output_image_name + ".gif",
    )
