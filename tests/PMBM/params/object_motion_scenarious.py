import numpy as np

from mot.common import Gaussian
from mot.configs import Object


# TODO single static object (no motion
single_static_object = [
    Object(
        initial=Gaussian(x=np.array([10.0, 10.0, 0.0, 0.0]), P=np.eye(4)),
        t_birth=0,
        t_death=69,
    )
]

# TODO two static objects (no motion)
two_static_objects = [
    Object(
        initial=Gaussian(x=np.array([-100.0, 100.0, 0.0, 0.0]), P=np.eye(4)),
        t_birth=0,
        t_death=69,
    ),
    Object(
        initial=Gaussian(x=np.array([100.0, 100.0, 0.0, 0.0]), P=np.eye(4)),
        t_birth=0,
        t_death=69,
    ),
]

# TODO three static objects (no motion)
tree_static_objects = [
    Object(
        initial=Gaussian(x=np.array([-250.0, 250.0, 0.0, 0.0]), P=np.eye(4)),
        t_birth=0,
        t_death=69,
    ),
    Object(
        initial=Gaussian(x=np.array([250.0, 250.0, 0.0, 0.0]), P=np.eye(4)),
        t_birth=0,
        t_death=69,
    ),
    Object(
        initial=Gaussian(x=np.array([0.0, -250.0, 0.0, 0.0]), P=np.eye(4)),
        t_birth=0,
        t_death=69,
    ),
]
# TODO single_object_linear_motion
single_object_linear_motion = [
    Object(
        initial=Gaussian(x=np.array([0.0, 0.0, 10.0, 10.0]), P=np.eye(4)),
        t_birth=0,
        t_death=100,
    )
]

# TODO two objects linear motion starting simulateneously
two_objects_linear_motion = [
    Object(
        initial=Gaussian(x=np.array([-100.0, 10.0, 10.0, 10.0]), P=np.eye(4)),
        t_birth=0,
        t_death=69,
    ),
    Object(
        initial=Gaussian(x=np.array([10.0, -10.0, 0.0, -10.0]), P=np.eye(4)),
        t_birth=0,
        t_death=69,
    ),
]

# TODO two objects linear motion starting - one after another
two_objects_linear_motion_delayed = [
    Object(
        initial=Gaussian(x=np.array([10.0, 10.0, 0.0, 10.0]), P=np.eye(4)),
        t_birth=5,
        t_death=69,
    ),
    Object(
        initial=Gaussian(x=np.array([10.0, -10.0, 0.0, -10.0]), P=np.eye(4)),
        t_birth=0,
        t_death=69,
    ),
]

# TODO big scenario linear motion
many_objects_linear_motion_delayed = [
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
