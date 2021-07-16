from dataclasses import dataclass, field
from typing import List

import numpy as np

from mot.configs import Object

from .initial_conditions import (
    linear_big_params,
    linear_n_mot_object_life_params,
    linear_sot_object_life_params,
    nonlinear_n_mot_object_life_params,
    nonlinear_sot_object_life_params,
)


@dataclass
class ScenarioConfig:
    n_births: int
    total_time: int
    object_configs: List[Object]
    dt: float
    P_D: float
    lambda_c: float
    range_c: np.ndarray
    sigma_r: float
    object_data_noisy: bool
    sigma_q: float = None
    sigma_V: float = 0.0
    sigma_omega: float = 0.0
    sigma_b: float = 0.0
    sensor_pos: List = field(default_factory=lambda: np.array([0.0, 0.0]))
    P_S: float = None


linear_sot = ScenarioConfig(
    n_births=1,
    total_time=100,
    object_configs=linear_sot_object_life_params,
    dt=1.0,
    sigma_q=5.0,
    P_D=0.9,
    lambda_c=5.0,
    range_c=np.array([[-1000, 1000], [-1000, 1000]]),
    sigma_r=10.0,
    object_data_noisy=False,
)

nonlinear_sot = ScenarioConfig(
    n_births=1,
    total_time=100,
    object_configs=nonlinear_sot_object_life_params,
    dt=1.0,
    sigma_V=1.0,
    sigma_omega=0.1 * np.pi / 180,
    P_D=0.9,
    lambda_c=10.0,
    range_c=np.array([[-1000, 1000], [-np.pi, np.pi]]),
    sigma_r=5.0,
    object_data_noisy=False,
    sigma_b=np.pi / 180,
    sensor_pos=np.array([300.0, 400.0]),
)

linear_n_mot = ScenarioConfig(
    n_births=5,
    total_time=100,
    object_configs=linear_n_mot_object_life_params,
    dt=1.0,
    sigma_q=5.0,
    P_D=0.9,
    lambda_c=10.0,
    range_c=np.array([[-1000, 1000], [-1000, 1000]]),
    sigma_r=10.0,
    object_data_noisy=False,
)

nonlinear_n_mot = ScenarioConfig(
    n_births=1,
    total_time=100,
    object_configs=nonlinear_n_mot_object_life_params,
    dt=1.0,
    sigma_q=5.0,
    P_D=0.9,
    lambda_c=10.0,
    range_c=np.array([[0, 2000], [-np.pi, np.pi]]),
    sigma_r=10.0,
    object_data_noisy=False,
    sigma_b=np.pi / 180,
    sensor_pos=np.array([300.0, 400.0]),
)

linear_full_mot = ScenarioConfig(
    n_births=12,
    total_time=100,
    object_configs=linear_big_params,
    dt=1.0,
    P_D=0.98,
    lambda_c=5.0,
    P_S=0.99,
    range_c=np.array([[-1000, 1000], [-1000, 1000]]),
    sigma_r=10.0,
    object_data_noisy=False,
    sigma_q=5.0,
)
