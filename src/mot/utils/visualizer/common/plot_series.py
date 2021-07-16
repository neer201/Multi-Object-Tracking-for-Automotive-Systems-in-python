import logging
from functools import singledispatch

import colorcet
import numpy as np
from matplotlib.lines import Line2D

from mot.common.state import Gaussian
from mot.simulator import MeasurementData, ObjectData
from mot.utils.visualizer.common.plot_primitives import BasicPlotter


CLUTTER_COLOR = colorcet.glasbey_category10[3]  # red
CLUTTER_MARKER = "+"
OBJECT_COLORS = colorcet.glasbey_category10[4:]
OBJECT_MARKER = "s"
OBJECT_MEASUREMENT_MARKER = "o"
OBJECT_MEASUREMENT_COLOR = "b"

logging.getLogger("matplotlib").setLevel(logging.WARNING)


def plot_measurement_scene(
    ax,
    measurements,
    timestep,
    clutter_color=CLUTTER_COLOR,
    clutter_marker=CLUTTER_MARKER,
):
    generated_by_clutter = measurements.clutter_data[timestep][0]
    artists = []
    for clutter_point in generated_by_clutter:
        artists.append(
            BasicPlotter.plot_point(
                x=clutter_point.squeeze()[0],
                y=clutter_point.squeeze()[1],
                ax=ax,
                color=clutter_color,
                marker=clutter_marker,
            )
        )

    for observation_point in measurements.meas_data[timestep]:
        artists.append(
            BasicPlotter.plot_point(
                x=observation_point[0],
                y=observation_point[1],
                ax=ax,
                color=OBJECT_MEASUREMENT_COLOR,
                marker=OBJECT_MEASUREMENT_MARKER,
            )
        )
    return artists


@singledispatch
def plot_series(series, ax, *args, **kwargs):
    raise NotImplementedError


@plot_series.register(ObjectData)
def _plot_series(series: ObjectData, ax, *args, **kwargs):
    for timestep in range(len(series)):
        objects_in_scene = series[timestep]
        for object_id in objects_in_scene.keys():
            state = objects_in_scene[object_id]
            curr_color, curr_marker = (
                OBJECT_COLORS[object_id],
                OBJECT_MARKER,
            )
            point, arrow, ellipse = BasicPlotter.plot_state(
                state,
                ax=ax,
                color=curr_color,
                center_marker=curr_marker,
            )
            ax.add_artist(point)

    legend_elements, labels = ax.get_legend_handles_labels()
    legend_elements.extend(
        [
            Line2D(
                [0],
                [0],
                marker=OBJECT_MARKER,
                color=OBJECT_COLORS[object_id],
                label=f"object {object_id}",
            )
            for object_id in range(series._ground_truth_config.n_births)
        ]
    )
    lgd = ax.legend(handles=legend_elements, loc="best", bbox_to_anchor=(1, 0.815))  # noqa
    return ax


@plot_series.register(MeasurementData)
def __plot_series(series: MeasurementData, ax, *args, **kwargs):
    for timestep in range(len(series)):
        plot_measurement_scene(ax, series, timestep)

    legend_elements, labels = ax.get_legend_handles_labels()
    legend_elements.append(Line2D([0], [0], marker=CLUTTER_MARKER, color=CLUTTER_COLOR, label="clutter"))
    lgd = ax.legend(handles=legend_elements, loc="best", bbox_to_anchor=(1, 0.815))  # noqa
    return ax


@plot_series.register(list)
def ___plot_series(series: list, ax, *args, **kwargs):
    for timestep in range(len(series)):
        scene = series[timestep]
        if isinstance(scene, Gaussian):
            for patch in BasicPlotter.plot_state(scene, ax=ax, color="m"):
                ax.add_artist(patch)

        elif isinstance(scene, np.ndarray):
            if len(scene) > 0:
                ax.add_artist(
                    BasicPlotter.plot_point(
                        x=scene.squeeze()[0],
                        y=scene.squeeze()[1],
                        ax=ax,
                        color="r",
                        marker="x",
                    )
                )
        elif isinstance(scene, list):
            if scene:
                for curr_object in scene:
                    if isinstance(curr_object, Gaussian):
                        for patch in BasicPlotter.plot_state(curr_object, ax=ax, color="m"):
                            ax.add_artist(patch)
    return ax


# @plot_series.register(tuple, list)
# def ____plot_series(series: tuple, ax, *args, **kwargs):
#     for timestep in range(len(series)):
#         object_ = series[timestep]
#         state = object_

#         print(state)
#         if not state:
#             print("not state")
#             return ax
#         for patch in BasicPlotter.plot_state(state, ax=ax, color="m"):
#             ax.add_artist(patch)

#     return ax


@plot_series.register(np.ndarray)
def ____plot_series(series: np.ndarray, ax, *args, **kwargs):
    for timestep in range(len(series)):
        states = series[timestep]
        for state in states:
            ax.add_artist(BasicPlotter.plot_point(x=state[0], y=state[1], ax=ax, color="m"))
    return ax
