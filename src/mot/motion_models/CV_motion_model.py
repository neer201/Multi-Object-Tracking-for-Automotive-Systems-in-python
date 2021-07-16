import numpy as np

from mot.common.state import Gaussian
from mot.motion_models import MotionModel


class ConstantVelocityMotionModel(MotionModel):
    def __init__(self, dt: float, sigma_q: float, *args, **kwargs):
        """Creates a 2D nearly constant velocity model

        Note:
            the motion model assumes that the state vector x consists of the following states:
            px -> X position
            py -> Y position
            vx -> X velocity
            vy -> Y velocity

        Attributes:
            d (scalar): object state dimension
            F (2 x 2 matrix): function handle return a motion transition function
            Q (4 x 4 matrix): motion noise covariance
            f (4 x 1 vector): function handle return state prediction

        Args:
            T (scalar): sampling time
            sigma (scalar): standart deviation of motion noise
        """

        self.d = 4
        self.dt = dt
        self.sigma = sigma_q
        super(ConstantVelocityMotionModel, self).__init__(*args, **kwargs)

    def f(self, state_vector: np.ndarray, dt: float):
        # TODO assert on state
        return self._transition_matrix(dt=dt) @ state_vector

    def F(self, state_vector: np.ndarray, dt: float):
        return self._transition_matrix(dt=dt)

    def Q(self, dt):
        return self._get_motion_noise_covariance(dt=dt, sigma=self.sigma)

    def move(self, state: Gaussian, dt: float = None, if_noisy: bool = False) -> Gaussian:
        dt = self.dt if dt is None else dt
        assert isinstance(dt, float)
        assert isinstance(state, Gaussian), f"Argument of wrong type! Type ={type}"
        if if_noisy:
            next_state = Gaussian(
                x=self._generator.multivariate_normal(
                    mean=self._transition_matrix(dt=dt) @ state.x,
                    cov=self._get_motion_noise_covariance(dt=dt),
                ),
                P=self._get_motion_noise_covariance(),
            )
        else:
            next_state = Gaussian(
                x=self._transition_matrix(dt=dt) @ state.x,
                P=self._get_motion_noise_covariance(),
            )
        return next_state

    def _transition_matrix(self, dt=None):
        dt = self.dt if dt is None else dt
        assert isinstance(dt, float)
        F = np.array(
            [
                [1.0, 0.0, dt, 0.0],
                [0.0, 1.0, 0.0, dt],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        return F

    def _get_motion_noise_covariance(self, dt=None, sigma=None):
        dt = self.dt if dt is None else dt
        assert isinstance(dt, float)

        sigma = self.sigma if sigma is None else sigma
        assert isinstance(sigma, float)

        Q = sigma ** 2 * np.array(
            [
                [dt ** 4 / 4.0, 0.0, dt ** 3 / 2.0, 0.0],
                [0.0, dt ** 4.0 / 4.0, 0.0, dt ** 3.0 / 2.0],
                [dt ** 3.0 / 2.0, 0.0, dt ** 2.0, 0.0],
                [0.0, dt ** 3.0 / 2.0, 0.0, dt ** 2.0],
            ]
        )
        return Q

    def __repr__(self) -> str:
        return f"Constant velocity motion model with dt = {self.dt} siqma = {self.sigma} "
