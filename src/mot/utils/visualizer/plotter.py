import logging
from typing import List, Union

import matplotlib.pyplot as plt
import numpy as np

from mot.simulator import MeasurementData, ObjectData
from mot.utils.visualizer.common.autoscale import autoscale
from mot.utils.visualizer.common.common import create_figure, set_mpl_params
from mot.utils.visualizer.common.plot_series import plot_series


set_mpl_params()

logging.getLogger("matplotlib").setLevel(logging.WARNING)


class Plot:
    def __init__(
        self,
        ax=None,
        title=None,
        out_path=None,
        show=False,
        is_autoscale=True,
        autoscale_margin=1.0,
        lim_x=(-1100, 1100),
        lim_y=(-1100, 1100),
        **kwargs,
    ):
        set_mpl_params()
        if ax is None:
            self.fig, self.ax = create_figure(title=title, **kwargs)
        else:
            self.ax = ax
        self.out_path = out_path
        self.show = show
        self.is_autoscale = is_autoscale
        self.autoscale_margin = autoscale_margin
        self.ax.set_xlim(lim_x)
        self.ax.set_ylim(lim_y)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        if self.is_autoscale:
            self.ax = autoscale(self.ax, "y", margin=self.autoscale_margin)
            self.ax = autoscale(self.ax, "x", margin=self.autoscale_margin)
            plt.tight_layout()

        if self.out_path is not None:
            plt.savefig(self.out_path)

        if self.show is not False:
            plt.show()


class Plotter:
    @staticmethod
    def plot_several(
        data: List,
        title=None,
        ax=None,
        out_path=None,
        show=False,
        is_autoscale=False,
        *args,
        **kwargs,
    ):
        assert isinstance(data, list)

        with Plot(
            ax=ax,
            title=title,
            out_path=out_path,
            show=show,
            autoscale_margin=100.0,
            is_autoscale=is_autoscale,
            **kwargs,
        ) as p:
            for series in data:
                plot_series(series, p.ax)
        return p.ax

    @staticmethod
    def plot(
        data: Union[MeasurementData, ObjectData, np.ndarray, List],
        title=None,
        ax=None,
        out_path=None,
        show=False,
        *args,
        **kwargs,
    ):

        with Plot(ax=ax, title=title, out_path=out_path, show=show, autoscale_margin=0.1, **kwargs) as p:
            plot_series(data, p.ax)
        return p.ax
