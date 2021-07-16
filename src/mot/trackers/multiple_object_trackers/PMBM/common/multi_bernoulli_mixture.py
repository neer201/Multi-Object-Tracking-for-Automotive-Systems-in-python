import logging
from collections import defaultdict
from typing import List

import numpy as np

# TODO make it ok
from .....common import normalize_log_weights
from .....measurement_models import MeasurementModel
from .....motion_models import MotionModel
from .global_hypothesis import GlobalHypothesis
from .track import Track


class MultiBernouilliMixture:
    """Track oriented approach using."""

    def __init__(self):
        self.tracks = {}
        self.global_hypotheses: List[GlobalHypothesis] = []

    def __repr__(self) -> str:
        return (
            self.__class__.__name__
            + " "
            + (f"num of tracks= {len(self.tracks)}, " f"num global hypotheses={len(self.global_hypotheses)}, ")
        )

    def add_track(self, track: Track):
        assert not (track.track_id in self.tracks.keys())
        self.tracks[track.track_id] = track

    def estimator(self, existense_probability_threshold, track_history_length_threshold):
        """Simply return objects set based on most probable global hypo."""
        if not self.global_hypotheses:
            logging.debug("Pool of global hypotheses is empty!")
            return None
        else:

            most_probable_global_hypo = max(self.global_hypotheses, key=lambda x: x.log_weight)
            logging.debug(f"most probable global hypothesis: {most_probable_global_hypo}")

        object_list = []  # list of {'object_id':'object_state'}
        logging.debug("\n estimations")
        for (track_id, sth_id) in most_probable_global_hypo.associations:
            if (
                self.tracks[track_id].single_target_hypotheses[sth_id].bernoulli.existence_probability
                > existense_probability_threshold
                and sth_id > track_history_length_threshold
            ):
                object_state = self.tracks[track_id].single_target_hypotheses[sth_id].bernoulli.state.x
                object_list.append({track_id: object_state})
                logging.debug(
                    f"r = {self.tracks[track_id].single_target_hypotheses[sth_id].bernoulli.existence_probability}, \
                    track id={track_id}, sth_id={sth_id}, state={object_state}"
                )

        return object_list

    def predict(
        self,
        motion_model: MotionModel,
        survival_probability: float,
        density_handler,
        dt: float,
    ) -> None:
        # MBM predict
        for track in self.tracks.values():
            for sth in track.single_target_hypotheses.values():
                sth.bernoulli.predict(
                    motion_model,
                    survival_probability,
                    density_handler,
                    dt,
                )

    def gating(self, z: np.ndarray, density_handler, meas_model: MeasurementModel, gating_size):

        gating_matrix = defaultdict(lambda: defaultdict(lambda: False))
        used_measurement_detected_indices = np.full(shape=[len(z)], fill_value=False)

        for track_id, track in self.tracks.items():
            for sth_id, sth in track.single_target_hypotheses.items():
                gating_matrix[track_id][sth_id][1] = density_handler.ellipsoidal_gating(
                    sth.bernoulli.state, z, meas_model, gating_size
                )
                used_measurement_detected_indices = np.logical_or(
                    used_measurement_detected_indices,
                    gating_matrix[track_id][sth_id],
                )

        return (gating_matrix, used_measurement_detected_indices)

    def update(
        self,
        detection_probability: float,
        measurements: np.ndarray,
        meas_model,
        density,
    ) -> None:
        """For each Bernoulli state density,
        create a misdetection hypothesis (Bernoulli component), and
        m object detection hypothesis (Bernoulli component),
        where m is the number of detections inside the ellipsoidal gate of the given state density.

        used_meas_indices : (hyp_tree_idx x bern_idx x meas_idx)
            [description]
        """
        logging.debug("\n Creating new STH in MBM")
        for track in self.tracks.values():
            for sth in track.single_target_hypotheses.values():
                logging.debug(f"\n For hypothesis: track_id={track.track_id} sth_id={sth.sth_id} {sth}")

                sth.missdetection_hypothesis = sth.create_missdetection_hypothesis(
                    detection_probability, track.get_new_sth_id()
                )
                logging.debug(f"Created missdetection hypothesis {sth}")

                # sth.detection_hypotheses = {
                #     meas_idx: sth.create_detection_hypothesis(
                #         measurements[meas_idx],
                #         detection_probability,
                #         meas_model,
                #         density,
                #         next(track.sth_id_generator),
                #     )
                #     for meas_idx in range(len(measurements))
                # }
                # copied_sth = copy.deepcopy(sth)
                sth.detection_hypotheses = sth.create_detection_hypotheses(
                    measurements,
                    detection_probability,
                    meas_model,
                    density,
                    [track.get_new_sth_id() for i in range(len(measurements))],
                )
                # from pprint import pprint
                # pprint([(
                #     sth.detection_hypotheses[idx],
                #     detection_hypotheses[idx],
                #     None,
                # ) for idx in range(len(measurements))])

                # for meas_idx, detection_hypothesis in sth.detection_hypotheses.items():
                #     logging.debug(
                #         f"Created detection hypothesis for meas_idx={meas_idx} STH = {detection_hypothesis}"
                #     )

    def prune_global_hypotheses(self, log_threshold: float) -> None:
        """Removes Bernoulli components with small probability of existence and reindex the hypothesis table.
        If a track contains no single object hypothesis after pruning, this track is removed

        good choice threshold : np.log(0.05)
        """
        self.global_hypotheses = [
            global_hypothesis
            for global_hypothesis in self.global_hypotheses
            if global_hypothesis.log_weight > log_threshold
        ]
        self.normalize_global_hypotheses_weights()

    def cap_global_hypothesis(self, max_number_of_global_hypothesis: int = 50):
        if len(self.global_hypotheses) > max_number_of_global_hypothesis:
            sorted_global_hypotheses = sorted(self.global_hypotheses, key=lambda x: x.log_weight)
            self.global_hypotheses = sorted_global_hypotheses[:max_number_of_global_hypothesis]
            self.normalize_global_hypotheses_weights()

    def prune_tree(self):
        used_associations = defaultdict(set)
        for global_hypothesis in self.global_hypotheses:
            for (track_id, sth_id) in global_hypothesis.associations:
                used_associations[track_id] |= set([sth_id])

        for track_id, track in self.tracks.items():
            track.single_target_hypotheses = {
                sth_id: track.single_target_hypotheses[sth_id] for sth_id in used_associations[track_id]
            }

    def remove_unused_tracks(self):
        used_associations = defaultdict(set)
        for global_hypothesis in self.global_hypotheses:
            for (track_id, sth_id) in global_hypothesis.associations:
                used_associations[track_id] |= set([sth_id])
        self.tracks = {
            track_id: track for (track_id, track) in self.tracks.items() if track_id in used_associations.keys()
        }

    def remove_unused_bernoullies(self):
        used_associations = defaultdict(set)
        for global_hypothesis in self.global_hypotheses:
            for (track_id, sth_id) in global_hypothesis.associations:
                used_associations[track_id] |= set([sth_id])
        for track_id, track in self.tracks.items():
            track.single_target_hypotheses = {
                sth_id: sth
                for (sth_id, sth) in track.single_target_hypotheses.items()
                if sth_id in used_associations[track_id]
            }

    def normalize_global_hypotheses_weights(self) -> None:
        global_hypo_log_w_unnorm = [global_hypothesis.log_weight for global_hypothesis in self.global_hypotheses]
        global_hypo_log_w_norm, _ = normalize_log_weights(global_hypo_log_w_unnorm)

        for global_hypo, normalized_log_weight in zip(self.global_hypotheses, global_hypo_log_w_norm):
            global_hypo.log_weight = normalized_log_weight
