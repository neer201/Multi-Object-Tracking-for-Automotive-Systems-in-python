import logging

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from mot.simulator.measurement_data_generator import MeasurementData
from mot.utils.visualizer.common.common import create_figure, set_mpl_params
from mot.utils.visualizer.common.plot_primitives import BasicPlotter


set_mpl_params()
plt.set_loglevel("info")

logging.getLogger("matplotlib").setLevel(logging.WARNING)


class Animator:
    @staticmethod
    def animate(timeseries_list, ax=None, title=None, filename=None, show=False, **kwargs):
        if ax is None:
            fig, ax = create_figure(title=title)

        # ax.set_xlim(-125, 1000)
        # ax.set_ylim(-125, 1000)
        timeseries_len = len(timeseries_list[0])

        global lines

        def update_frame(timestep):
            scene = []
            for timeseries in timeseries_list:
                #     if type(timeseries) is ObjectData:
                #         for object_id in timeseries[timestep].keys():
                #             scene.extend(
                #                 BasicPlotter.plot_state(
                #                     timeseries[timestep][object_id],
                #                     ax=ax,
                #                     color="r",
                #                     center_marker="x",
                #                 )
                #             )

                # # TODO add clutter and ground truth
                if isinstance(timeseries, MeasurementData):
                    measurements = timeseries[timestep]

                    scene.append(
                        BasicPlotter.plot_point(
                            x=measurements[:, 0],
                            y=measurements[:, 1],
                            ax=ax,
                            color="r",
                            marker="*",
                            label="measurements",
                        )
                    )
                # elif isinstance(timeseries, tuple) or isinstance(timeseries, list):
                #     if not isinstance(timeseries[timestep], list):
                #         if isinstance(timeseries[timestep], Gaussian):
                #             scene.extend(
                #                 BasicPlotter.plot_state(
                #                     timeseries[timestep], ax=ax, color="g"
                #                 )
                #             )
                #         elif isinstance(timeseries[0], np.ndarray):
                #             scene.append(
                #                 BasicPlotter.plot_point(
                #                     x=timeseries[timestep][0],
                #                     y=timeseries[timestep][1],
                #                     ax=ax,
                #                     color="g",
                #                 )
                #             )
                #     else:
                #         for object_ in timeseries[timestep]:
                #             scene.extend(
                #                 BasicPlotter.plot_state(object_, ax=ax, color="g")
                #             )
                # else:
                #     assert False
            # autoscale(ax, "y", margin=0.1)
            # autoscale(ax, "x", margin=0.1)

            # plt.tight_layout()
            return scene

        ani = FuncAnimation(
            fig=fig,
            func=update_frame,
            frames=list(range(timeseries_len)),
            interval=10,
            save_count=500,
            blit=True,
            repeat=False,
        )

        if show is not False:
            plt.show()

        if filename is not None:
            ani.save(filename, writer=matplotlib.animation.FFMpegWriter())
