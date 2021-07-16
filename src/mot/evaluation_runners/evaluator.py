from collections import defaultdict

import matplotlib.pyplot as plt
import motmetrics
import numpy as np

from mot.metrics import GOSPA
from mot.utils import get_images_dir
from mot.utils.visualizer.common.plot_series import OBJECT_COLORS as object_colors


class OneSceneMOTevaluator:
    def __init__(self):
        # scene data: a list of samples
        self.estimates = []
        self.measurements = []
        self.gt = []

        # metrics
        self.gospa_metrics = []
        self.mot_metric_accumulator = motmetrics.MOTAccumulator()
        self.scene_metric = {}

    def step(self, sample_measurements, sample_estimates, sample_gt, timestep):
        self.estimates.append(sample_estimates)
        self.measurements.append(sample_measurements)
        self.gt.append(sample_gt)
        target_points = np.array([target.x[:2] for target in sample_gt.values()])
        target_ids = [target_id for target_id in sample_gt.keys()]
        # target_points = []
        # target_ids = []
        # for target_id, target_coords in sample_gt.items():
        #     target_ids.append(target_id)
        #     target_points.append(target_coords[:2])
        # target_points = np.array(target_points)

        if sample_estimates:
            estimation_points = np.array([list(estimation.values())[0][:2] for estimation in sample_estimates])
            estimation_ids = [estimation_ids for estimation in sample_estimates for estimation_ids in estimation.keys()]
        else:
            estimation_points = np.array([])
            estimation_ids = []

        sample_GOSPA = GOSPA(target_points, estimation_points)
        self.gospa_metrics.append(sample_GOSPA)

        distance_matrix = motmetrics.distances.norm2squared_matrix(target_points, estimation_points)
        self.mot_metric_accumulator.update(target_ids, estimation_ids, dists=distance_matrix, frameid=timestep)

    def post_processing(self):
        # pass
        # self._metrics_calculation_over_scene()
        self._plot_result()

    def _metrics_calculation_over_scene(self):
        self.scene_metric["rms_gospa"] = np.sqrt(np.mean(np.power(np.array(self.gospa_metrics), 2)))

        mh = motmetrics.metrics.create()
        summary = mh.compute(
            self.mot_metric_accumulator,
            metrics=["num_frames", "mota", "motp", "idp"],
            name="acc",
        )
        self.scene_metric["mota"] = summary["mota"].item()
        self.scene_metric["motp"] = summary["mota"].item()
        self.scene_metric["idp"] = summary["mota"].item()

    def _plot_result(self):
        fig, axs = plt.subplots(2, 5, figsize=(8 * 5, 8 * 2), sharey=False, sharex=False)
        simulation_steps = len(self.measurements)
        axs[0, 0].grid(which="both", linestyle="-", alpha=0.5)
        axs[0, 0].set_title(label="ground truth")
        axs[0, 0].set_xlabel("x position")
        axs[0, 0].set_ylabel("y position")
        for timestep in range(simulation_steps):
            objects_in_scene = self.gt[timestep]
            for object_id in objects_in_scene.keys():
                state = objects_in_scene[object_id]
                gt_pos_x, gt_pos_y = state.x[:2]
                axs[0, 0].scatter(gt_pos_x, gt_pos_y, color=object_colors[object_id % 252])

        # axs[0, 0] = Plotter.plot_several(
        #     [self.gt],
        #     ax=axs[0, 0],
        #     out_path=None,
        #     is_autoscale=False,
        # )
        axs[1, 0].get_shared_x_axes().join(axs[1, 0], axs[0, 0])
        axs[1, 0].get_shared_y_axes().join(axs[1, 0], axs[0, 0])
        axs[1, 0].grid(which="both", linestyle="-", alpha=0.5)
        axs[1, 0].set_title(label="measurements")
        axs[1, 0].set_xlabel("x position")
        axs[1, 0].set_ylabel("y position")
        for sample_measurements in self.measurements:
            if len(sample_measurements) > 0:
                axs[1, 0].scatter(
                    sample_measurements.states[:, 0], sample_measurements.states[:, 1], color="r", marker="x"
                )

        # axs[1, 0] = Plotter.plot_several(
        # [self.measurements],
        # ax=axs[1, 0],
        # out_path=None,
        # is_autoscale=False,
        # )
        axs[0, 1].get_shared_x_axes().join(axs[0, 1], axs[0, 0])
        axs[0, 1].get_shared_y_axes().join(axs[0, 1], axs[0, 0])
        axs[0, 1].grid(which="both", linestyle="-", alpha=0.5)
        axs[0, 1].set_title(label="estimations")
        # axs[0, 1].set_xlim([-1100, 1100])
        # axs[0, 1].set_ylim([-1100, 1100])
        axs[0, 1].set_xlabel("x position")
        axs[0, 1].set_ylabel("y position")

        axs[0, 2].grid(which="both", linestyle="-", alpha=0.5)
        axs[0, 2].set_title(label="estimated x position over time")
        axs[0, 2].set_xlabel("time")
        axs[0, 2].set_ylabel("x position")
        axs[0, 2].set_xlim([0, simulation_steps])
        axs[0, 2].set_xticks(np.arange(0, simulation_steps, step=int(simulation_steps / 10)))

        axs[1, 2].grid(which="both", linestyle="-", alpha=0.5)
        axs[1, 2].set_title(label="estimated y position over time")
        axs[1, 2].set_xlabel("time")
        axs[1, 2].set_ylabel("y position")
        axs[1, 2].set_xlim([0, simulation_steps])
        axs[1, 2].set_xticks(np.arange(0, simulation_steps, step=int(simulation_steps / 10)))

        axs[0, 3].get_shared_y_axes().join(axs[0, 3], axs[0, 2])
        axs[0, 3].grid(which="both", linestyle="-", alpha=0.5)
        axs[0, 3].set_title(label="GT x position over time")
        axs[0, 3].set_xlabel("time")
        axs[0, 3].set_ylabel("x position")
        axs[0, 3].set_xlim([0, simulation_steps])
        axs[0, 3].set_xticks(np.arange(0, simulation_steps, step=int(simulation_steps / 10)))

        axs[0, 3].get_shared_y_axes().join(axs[1, 3], axs[1, 2])
        axs[1, 3].grid(which="both", linestyle="-", alpha=0.5)
        axs[1, 3].set_title(label="GT y position over time")
        axs[1, 3].set_xlabel("time")
        axs[1, 3].set_ylabel("y position")
        axs[1, 3].set_xlim([0, simulation_steps])
        axs[1, 3].set_xticks(np.arange(0, simulation_steps, step=int(simulation_steps / 10)))

        for timestep in range(simulation_steps):
            objects_in_scene = self.gt[timestep]
            for object_id in objects_in_scene.keys():
                state = objects_in_scene[object_id]
                gt_pos_x, gt_pos_y = state.x[:2]
                axs[0, 3].scatter(timestep, gt_pos_x, color=object_colors[object_id % 252])
                axs[1, 3].scatter(timestep, gt_pos_y, color=object_colors[object_id % 252])

        axs[0, 4].get_shared_y_axes().join(axs[0, 4], axs[0, 3])
        axs[0, 4].grid(which="both", linestyle="-", alpha=0.5)
        axs[0, 4].set_title(label="x measurements over time")
        axs[0, 4].set_xlabel("time")
        axs[0, 4].set_ylabel("x position")
        axs[0, 4].set_xlim([0, simulation_steps])
        axs[0, 4].set_xticks(np.arange(0, simulation_steps, step=int(simulation_steps / 10)))

        axs[1, 4].get_shared_y_axes().join(axs[1, 3], axs[1, 2])
        axs[1, 4].grid(which="both", linestyle="-", alpha=0.5)
        axs[1, 4].set_title(label="y measurements over time")
        axs[1, 4].set_xlabel("time")
        axs[1, 4].set_ylabel("y position")
        axs[1, 4].set_xlim([0, simulation_steps])
        axs[1, 4].set_xticks(np.arange(0, simulation_steps, step=int(simulation_steps / 10)))

        for timestep in range(simulation_steps):
            objects_in_scene = self.gt[timestep]
            for measurement in self.measurements[timestep]:
                meas_x, meas_y = measurement.measurement
                axs[0, 4].scatter(timestep, meas_x, color="r", marker="+")
                axs[1, 4].scatter(timestep, meas_y, color="r", marker="+")

        lines = defaultdict(lambda: [])  # target_id: line
        timelines = defaultdict(lambda: [])
        for timestep, current_timestep_estimations in enumerate(self.estimates):
            if current_timestep_estimations:
                for estimation in current_timestep_estimations:
                    for target_id, state_vector in estimation.items():
                        pos_x, pos_y = state_vector[:2]
                        lines[target_id].append((pos_x, pos_y))
                        axs[0, 2].scatter(timestep, pos_x, color=object_colors[target_id % 252])
                        axs[1, 2].scatter(timestep, pos_y, color=object_colors[target_id % 252])
                        timelines[target_id].append((timestep, pos_x, pos_y))

        for target_id, estimation_list in timelines.items():
            timesteps = [time for (time, _, _) in estimation_list]
            poses_x = [pos_x for (_, pos_x, _) in estimation_list]
            poses_y = [pos_y for (_, _, pos_y) in estimation_list]
            axs[0, 2].plot(timesteps, poses_x, color=object_colors[target_id % 252])
            axs[1, 2].plot(timesteps, poses_y, color=object_colors[target_id % 252])

        for target_id, estimation_list in lines.items():
            for (pos_x, pos_y) in estimation_list:
                axs[0, 1].scatter(pos_x, pos_y, color=object_colors[target_id % 252])

        # fig.suptitle(
        #     f"RMS GOSPA ={self.scene_metric['rms_gospa']:.1f} "
        #     f"MOTA ={self.scene_metric['mota']:.1f} "
        #     f"MOTP ={self.scene_metric['motp']:.1f} "
        #     f"IDP ={self.scene_metric['idp']:.1f} ",
        #     fontweight="bold",
        # )
        meta = f"{np.random.randint(100)}"
        plt.savefig(get_images_dir(__file__) + "/" + "results_" + meta + ".png")
