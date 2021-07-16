import pprint

from .single_target_hypothesis import SingleTargetHypothesis


class Track:
    """Represents a track - hypotheses tree.
    The root is associated with a unique target.
    Leaves are hypotheses which represent association of this target
    with corresponding measurements or misdetections.
    """

    max_track_id = 1000000
    current_idx = 0

    def get_id(cls):
        returned_idx = Track.current_idx
        Track.current_idx += 1
        return returned_idx

    def __init__(self, initial_sth=None):
        self.track_id = Track.get_id(Track)
        self.sth_id_counter = 0

        self.single_target_hypotheses = {self.get_new_sth_id(): initial_sth}

    def get_new_sth_id(self):
        returned_value = self.sth_id_counter
        self.sth_id_counter += 1
        return returned_value

    def add_sth(self, sth: SingleTargetHypothesis) -> None:
        self.single_target_hypotheses[self.get_new_sth_id()] = sth

    def __repr__(self) -> str:
        sth_rep = (
            f"STH: \n {pprint.pformat(self.single_target_hypotheses)}" if len(self.single_target_hypotheses) < 3 else ""
        )
        return (
            self.__class__.__name__
            + (f" id = {self.track_id}" f" number of sth = {len(self.single_target_hypotheses)} ")
            + sth_rep
        )

    @classmethod
    def from_sth(cls, single_target_hypo):
        return cls(initial_sth=single_target_hypo)

    def cut_tree(self):
        new_single_target_hypotheses = {}
        for parent_sth in self.single_target_hypotheses.values():
            for child_sth in parent_sth.detection_hypotheses.values():
                new_single_target_hypotheses[child_sth.sth_id] = child_sth
        self.single_target_hypotheses = new_single_target_hypotheses
