import numpy as np

from mot.common.state import Gaussian
from mot.configs import Object


linear_sot_object_life_params = [
    Object(
        initial=Gaussian(x=np.array([0.0, 0.0, 10.0, 10.0]), P=np.eye(4)),
        t_birth=0,
        t_death=100,
    )
]

nonlinear_sot_object_life_params = [
    Object(
        initial=Gaussian(
            x=np.array([0.0, 0.0, 10.0, 0.0, np.pi / 180.0]),
            P=np.diag(np.power([1.0, 1.0, 1.0, 1.0 * np.pi / 180.0, 1 * np.pi / 180], 2)),
        ),
        t_birth=0,
        t_death=400,
    )
]

linear_n_mot_object_life_params = [
    Object(
        initial=Gaussian(x=np.array([0.0, 0.0, 0.0, -10.0]), P=10 * np.eye(4)),
        t_birth=0,
        t_death=99,
    ),
    Object(
        initial=Gaussian(x=np.array([400.0, -600.0, -10.0, 5.0]), P=np.eye(4)),
        t_birth=0,
        t_death=99,
    ),
    Object(
        initial=Gaussian(x=np.array([-800.0, -200.0, 20.0, -5.0]), P=np.eye(4)),
        t_birth=0,
        t_death=99,
    ),
    Object(
        initial=Gaussian(x=np.array([0.0, 0.0, 7.5, -5.0]), P=np.eye(4)),
        t_birth=0,
        t_death=99,
    ),
    Object(
        initial=Gaussian(x=np.array([-200.0, 800.0, -3.0, -15.0]), P=np.eye(4)),
        t_birth=0,
        t_death=99,
    ),
]

# non-linear state vector (X-pos, Y-pos, velocity, heading, turn-rate)

default_nonlinear_P = np.diag(np.power([1e2, 1e2, 1e1, 1.0 * np.pi / 180.0, 1 * np.pi / 180], 2))
nonlinear_n_mot_object_life_params = [
    Object(
        initial=Gaussian(x=np.array([0.0, 0.0, 5.0, 0.0, np.pi / 180]), P=default_nonlinear_P),
        t_birth=0,
        t_death=99,
    ),
    Object(
        initial=Gaussian(x=np.array([20.0, -20.0, -20.0, 0.0, np.pi / 90]), P=default_nonlinear_P),
        t_birth=0,
        t_death=99,
    ),
    Object(
        initial=Gaussian(x=np.array([-20.0, 10.0, -10.0, 0.0, np.pi / 360]), P=default_nonlinear_P),
        t_birth=0,
        t_death=99,
    ),
    Object(
        initial=Gaussian(x=np.array([-10.0, 10.0, 8.0, 0.0, np.pi / 270]), P=default_nonlinear_P),
        t_birth=0,
        t_death=99,
    ),
]

linear_big_params = [
    Object(
        initial=Gaussian(x=np.array([0.0, 0.0, 0.0, -10.0]), P=np.eye(4)),
        t_birth=0,
        t_death=69,
    ),
    Object(
        initial=Gaussian(x=np.array([400.0, -600.0, -10.0, 5.0]), P=np.eye(4)),
        t_birth=0,
        t_death=99,
    ),
    Object(
        initial=Gaussian(x=np.array([-800.0, -200.0, 20.0, -5.0]), P=np.eye(4)),
        t_birth=0,
        t_death=69,
    ),
    Object(
        initial=Gaussian(x=np.array([400.0, -600.0, -7.0, -4.0]), P=np.eye(4)),
        t_birth=19,
        t_death=99,
    ),
    Object(
        initial=Gaussian(x=np.array([400.0, -600.0, -2.5, 10.0]), P=np.eye(4)),
        t_birth=19,
        t_death=99,
    ),
    Object(
        initial=Gaussian(x=np.array([0.0, 0.0, 7.5, -5.0]), P=np.eye(4)),
        t_birth=19,
        t_death=99,
    ),
    Object(
        initial=Gaussian(x=np.array([-800.0, -200.0, 12.0, 7.0]), P=np.eye(4)),
        t_birth=39,
        t_death=99,
    ),
    Object(
        initial=Gaussian(x=np.array([-200.0, 800.0, 15.0, -10.0]), P=np.eye(4)),
        t_birth=39,
        t_death=99,
    ),
    Object(
        initial=Gaussian(x=np.array([-800.0, -200.0, 3.0, 15.0]), P=np.eye(4)),
        t_birth=59,
        t_death=99,
    ),
    Object(
        initial=Gaussian(x=np.array([-200.0, 800.0, -3.0, -15.0]), P=np.eye(4)),
        t_birth=59,
        t_death=99,
    ),
    Object(
        initial=Gaussian(x=np.array([0.0, 0.0, -20.0, -15.0]), P=np.eye(4)),
        t_birth=79,
        t_death=99,
    ),
    Object(
        initial=Gaussian(x=np.array([-200.0, 800.0, 15.0, -5.0]), P=np.eye(4)),
        t_birth=79,
        t_death=99,
    ),
]
