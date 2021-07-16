import logging

import numpy as np
from matplotlib.patches import Ellipse, FancyArrow

from mot.common.state import Gaussian


logging.getLogger("matplotlib").setLevel(logging.WARNING)


class BasicPlotter:
    @staticmethod
    def plot_point(
        ax,
        x,
        y,
        label=None,
        marker="o",
        color="b",
        marker_size=50,
    ):
        scatter = ax.scatter(x, y, marker=marker, color=color, label=label, s=marker_size, edgecolors="k")
        return scatter

    @staticmethod
    def plot_covariance_ellipse(ax, mean, covariance, color="b"):
        assert mean.shape == (2,), f"mean has {mean.shape} shape"
        covariance = covariance[:2, :2]
        assert covariance.shape == (2, 2), f"covariance has {covariance.shape} shape"
        lambda_, v = np.linalg.eig(covariance)
        lambda_ = np.sqrt(lambda_)
        ell_width, ell_height = lambda_[0] * 2, lambda_[1] * 2
        ell_angle = np.rad2deg(np.arccos(v[0, 0]))
        ellipse = Ellipse(
            xy=mean,
            width=ell_width,
            height=ell_height,
            angle=ell_angle,
            alpha=0.3,
            zorder=5,
        )
        ellipse.set_edgecolor("k")
        ellipse.set_facecolor(color)
        return ellipse

    @staticmethod
    def plot_state(
        state: Gaussian,
        ax,
        color,
        center_marker="*",
        label=None,
    ):

        assert isinstance(state, Gaussian)
        pos_x, pos_y = state.x[0], state.x[1]

        # draw position
        point = BasicPlotter.plot_point(ax, x=pos_x, y=pos_y, color=color, marker=center_marker, label=label)

        # plot velocity vector
        if state.x.size > 2:
            arrow = FancyArrow(
                x=pos_x,
                y=pos_y,
                dx=state.x[2],
                dy=state.x[3],
                length_includes_head=True,
                head_length=2.0,
                head_width=1.0,
            )

        # plot covariance ellipse
        cov_ell = BasicPlotter.plot_covariance_ellipse(
            ax=ax, mean=np.array([pos_x, pos_y]), covariance=state.P, color=color
        )
        return point, arrow, cov_ell
