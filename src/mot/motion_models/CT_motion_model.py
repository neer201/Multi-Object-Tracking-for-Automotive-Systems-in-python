import numpy as np

from mot.common.state import Gaussian
from mot.motion_models import MotionModel


class CoordinateTurnMotionModel(MotionModel):
    def __init__(self, dt: float, sigma_V: float, sigma_omega: float, *args, **kwargs):
        """Creates a 2D coordinate turn model with nearly constant polar velocity and turn rate

        Note:
            the motion model assumes that the state vactor x consists of the following states:
            px -> X position
            py -> Y position
            v  -> velocity
            phi -> heading
            omega -> turn rate

        Attributes:
            d (scalar): object state dimension
            F (5 x 5 matrix): function handle return a motion transition function
            Q (5 x 5 matrix): motion noise covariance
            f (5 x 1 vector): function handle return state prediction

        Args:
            T (scalar): sampling time
            sigma_V (scalar): standart deviation of motion noise added to polar velocity
            sigma_omega (scalar): standard deviation of motion noise added to turn rate
        """
        self.d = 5
        self.dt = dt
        self.sigma_V = sigma_V
        self.sigma_omega = sigma_omega

    def __repr__(self) -> str:
        return self.__class__.__name__ + (
            f"(d={self.d}, " f"dt={self.dt}, " f"sigma_V={self.sigma_V}, " f"sigma_omega={self.sigma_omega}, "
        )

    def move(self, state: Gaussian, dt: float = None, if_noisy: bool = False) -> Gaussian:
        dt = self.dt if dt is None else dt
        assert isinstance(dt, float)
        assert isinstance(state, Gaussian), f"Argument of wrong type! Type ={type}"

        if if_noisy:
            next_state = Gaussian(
                x=self._generator.multivariate_normal(
                    mean=self._f(state, dt=dt),
                    cov=self.Q(dt=dt),
                ),
                P=self.get_motion_noise_covariance(),
            )
        else:
            next_state = Gaussian(
                x=self._f(state, dt=dt),
                P=self.Q(dt=dt),
            )
        return next_state

    def f(self, state_vector, dt):
        # TODO assert on state
        pos_x, pos_y, v, phi, omega = state_vector
        next_state = np.array([dt * v * np.cos(phi), dt * v * np.sin(phi), 0, dt * omega, 0])
        return state_vector + next_state

    def F(self, state_vector, dt):
        pos_x, pos_y, v, phi, omega = state_vector
        return np.array(
            [
                [1.0, 0.0, dt * np.cos(phi), -dt * v * np.sin(phi), 0],
                [0.0, 1.0, dt * np.sin(phi), dt * v * np.cos(phi), 0],
                [0.0, 0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0, dt],
                [0.0, 0.0, 0.0, 0.0, 1.0],
            ]
        )

    def _f(self, state, dt):
        pos_x, pos_y, v, phi, omega = state.x

        next_state = np.array(
            [
                pos_x + ((2 * v) / omega) * np.sin(omega * dt / 2) * np.cos(phi + omega * dt / 2),
                pos_y + ((2 * v) / omega) * np.sin(omega * dt / 2) * np.sin(phi + omega * dt / 2),
                v,
                phi + omega * dt,
                omega,
            ]
        )
        return next_state

    def Q(self, dt=None):
        """Motion noise covariance"""
        return np.array(
            [
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, self.sigma_V ** 2, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, self.sigma_omega ** 2],
            ]
        )  # yapf: disable


# def _transition_matrix(self, state, dt=None):
#     dt = self.dt if dt is None else dt
#     assert isinstance(dt, float)
#     F = np.array([1.0, 0.0,d ])
