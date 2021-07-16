import itertools
import logging as lg
from itertools import repeat
from multiprocessing import Pool
from typing import Dict

import numpy as np
import scipy
import scipy.stats

from mot.common import GaussianDensity, GaussianMixture, ObservationList
from mot.configs import SensorModelConfig
from mot.measurement_models import MeasurementModel
from mot.motion_models import MotionModel
from mot.trackers.multiple_object_trackers.PMBM.common import (
    AssignmentSolver,
    Association,
    BirthModel,
    GlobalHypothesis,
    MultiBernouilliMixture,
    PoissonRFS,
    assign,
)
from mot.utils.timer import Timer


def solve(f):
    return f.solve()


class PMBM:
    def __init__(
        self,
        meas_model: MeasurementModel,
        sensor_model: SensorModelConfig,
        motion_model: MotionModel,
        birth_model: GaussianMixture,
        max_number_of_hypotheses: int,
        gating_percentage: float,
        detection_probability: float,
        survival_probability: float,
        existense_probability_threshold: float,
        track_history_length_threshold: int,
        density: GaussianDensity,
        initial_PPP_intensity: GaussianMixture,
        *args,
        **kwargs,
    ):
        assert isinstance(meas_model, MeasurementModel)
        assert isinstance(sensor_model, SensorModelConfig)
        assert isinstance(motion_model, MotionModel)
        assert isinstance(birth_model, BirthModel)
        assert isinstance(max_number_of_hypotheses, int)
        assert isinstance(gating_percentage, float)
        assert isinstance(detection_probability, float)
        assert isinstance(survival_probability, float)
        assert issubclass(density, GaussianDensity)

        self.timestep = 0
        self.density = density
        self.sensor_model = sensor_model
        self.motion_model = motion_model
        self.birth_model = birth_model
        self.meas_model = meas_model

        # models death of an object (aka P_S)
        self.survival_probability = survival_probability

        # models detection (and missdetection) of an object (aka P_D)
        self.detection_probability = detection_probability

        # Interval for ellipsoidal gating (aka P_G)
        self.gating_percentage = gating_percentage
        self.gating_size = scipy.stats.chi2.ppf(self.gating_percentage, df=self.meas_model.d)

        self.max_number_of_hypotheses = max_number_of_hypotheses
        self.existense_probability_threshold = existense_probability_threshold
        self.track_history_length_threshold = track_history_length_threshold

        self.PPP = PoissonRFS(intensity=initial_PPP_intensity)
        self.MBM = MultiBernouilliMixture()
        self.assingner_pool = Pool(processes=6)

    def __repr__(self) -> str:
        return self.__class__.__name__ + (
            f"(current_timestep={self.timestep}, "
            f"MBM components ={len(self.MBM.tracks)}, "
            f"Global hypotheses={len(self.MBM.global_hypotheses)}, "
            f"PPP components={len(self.PPP.intensity)}, "
        )

    def step(self, measurements: ObservationList, dt: float, ego_pose: Dict = None):
        if ego_pose is None:
            ego_pose = {"translation": (0.0, 0.0, 0.0), "rotation": (0.0, 0.0, 0.0, 0.0)}
        self.increment_timestep()
        self.predict(
            self.birth_model.get_born_objects_intensity(params={"measurements": measurements, "ego_pose": ego_pose}),
            self.motion_model,
            self.survival_probability,
            self.density,
            dt,
        )
        self.update(measurements)
        estimates = self.estimator()
        self.reduction()
        return estimates

    def increment_timestep(self):
        self.timestep += 1

    @Timer(name="PMBM preiction step")
    def predict(
        self,
        birth_model: BirthModel,
        motion_model: MotionModel,
        survival_probability: float,
        density: GaussianDensity,
        dt: float,
    ) -> None:
        assert isinstance(birth_model, GaussianMixture)
        assert isinstance(motion_model, MotionModel)
        assert isinstance(survival_probability, float)

        self.MBM.predict(motion_model, survival_probability, density, dt)
        self.PPP.predict(motion_model, survival_probability, density, dt)
        self.PPP.birth(birth_model)

    @Timer(name="PMBM update step")
    def update(self, measurements: np.ndarray) -> None:

        if len(measurements) == 0:
            lg.debug("\n no measurements!")
            return
        lg.debug(f"\n===current timestep: {self.timestep}===")
        lg.debug(f"\n   Observable measurements: \n {measurements}")
        lg.debug(f"\n   global hypotheses {self.MBM.global_hypotheses}")
        lg.debug(f"\n   MBM tracks {self.MBM.tracks} \n")

        new_tracks = self.PPP.get_targets_detected_for_first_time(
            measurements,
            self.sensor_model.intensity_c,
            self.meas_model,
            self.detection_probability,
        )

        self.MBM.update(
            self.detection_probability,
            measurements,
            self.meas_model,
            self.density,
        )

        # Update of PPP intensity for undetected objects that remain undetected
        self.PPP.undetected_update(self.detection_probability)

        lg.debug(f"\n   new tracks {new_tracks} \n")
        lg.debug(f"\n   current PPP components \n {self.PPP.intensity}")

        if not self.MBM.global_hypotheses or not self.MBM.tracks:
            self.MBM.tracks.update(new_tracks)
            hypo_list = []
            for track_id, track in self.MBM.tracks.items():
                for sth_id, _sth in track.single_target_hypotheses.items():
                    hypo_list.append(Association(track_id, sth_id))

            self.MBM.global_hypotheses.append(GlobalHypothesis(log_weight=0.0, associations=hypo_list))

        else:
            parrallel_assignment = False
            if parrallel_assignment:
                parallel_global_hypo = self.assingner_pool.starmap(
                    assign,
                    zip(
                        self.MBM.global_hypotheses,
                        repeat(self.MBM.tracks),
                        repeat(new_tracks),
                        repeat(measurements),
                        repeat(self.max_number_of_hypotheses),
                    ),
                )
                # parallel_global_hypo = self.assingner_pool.map(solve, assignment_problems
                new_global_hypotheses = itertools.chain.from_iterable(parallel_global_hypo)
            else:
                assignment_problems = [
                    AssignmentSolver(
                        global_hypothesis=global_hypothesis,
                        old_tracks=self.MBM.tracks,
                        new_tracks=new_tracks,
                        measurements=measurements,
                        num_of_desired_hypotheses=self.max_number_of_hypotheses,
                    )
                    for global_hypothesis in self.MBM.global_hypotheses
                ]
                new_global_hypotheses = itertools.chain.from_iterable(
                    [problem.solve() for problem in assignment_problems]
                )

            with Timer(name="Prepation for the next step"):
                self.update_tree()
                self.MBM.tracks.update(new_tracks)
                self.MBM.global_hypotheses = list(new_global_hypotheses)
                self.MBM.normalize_global_hypotheses_weights()
                self.MBM.prune_tree()

    def update_tree(self):
        """1. Move children to upper lever."""
        for track in self.MBM.tracks.values():
            track.cut_tree()

    @Timer(name="PMBM estimation step")
    def estimator(self):
        estimates = self.MBM.estimator(
            existense_probability_threshold=self.existense_probability_threshold,
            track_history_length_threshold=self.track_history_length_threshold,
        )
        return estimates

    @Timer(name="PMBM reduction step")
    def reduction(self) -> None:
        self.PPP.prune(threshold=-15)
        self.MBM.prune_global_hypotheses(log_threshold=np.log(0.01))
        self.MBM.cap_global_hypothesis(self.max_number_of_hypotheses)
        self.MBM.remove_unused_tracks()
        self.MBM.remove_unused_bernoullies()
