from ..configs.ground_truth_config import GroundTruthConfig
from ..motion_models import MotionModel


class ObjectData:
    """Generate groundtruth object data"""

    def __init__(
        self,
        ground_truth_config: GroundTruthConfig,
        motion_model: MotionModel,
        if_noisy: bool,
    ):
        """Init generator

        Args:
            ground_truth (GroundTruth): specifies the parameters used to generate groundtruth
            motion_model (MotionModel): a structure specifies the motion model parameters
            if_noisy (bool): boolean value indicating whether to generate noisy object state
                             sequence or not

        Attributes:
            objectdata.X ((K x 1) cell array ) each cell stores object states of size
                     (object state dimenstion) x (number of objects at corresponding time step)
            objectdata.N ((K x 1) cell array ) each cell stores the number of objects
                                               at corresponding time step

        """
        self._ground_truth_config = ground_truth_config
        self._motion_model = motion_model
        self._if_noisy = if_noisy
        self.objects_state_data = self.generate_objects_data()

    def __len__(self):
        return self._ground_truth_config.total_time

    def __getitem__(self, key):
        return self.objects_state_data[key]

    def generate_objects_data(self):
        object_state_history = [{} for timestep in range(self._ground_truth_config.total_time)]
        for object_config in self._ground_truth_config.object_configs:
            state = object_config.initial_state
            for timestep in range(self._ground_truth_config.total_time):
                if timestep in range(object_config.t_birth, object_config.t_death):
                    object_state_history[timestep][object_config.id] = state
                    next_state = self._motion_model.move(state, if_noisy=self._if_noisy)
                    state = next_state
        return tuple(object_state_history)

    def __repr__(self) -> str:
        return self.__class__.__name__ + (
            f"(ground_truth_config={self._ground_truth_config}, "
            f"motion_model={self._motion_model}, "
            f"if_noisy={self._if_noisy}, "
            f"X={self.objects_state_data}, "
        )

    @property
    def data(self):
        """[summary]

        Returns
        -------
        Tuple : (tumesteps x objects_in_scene x object_state_dim)

        """
        # TODO
        SINGLE_OBJECT_KEY = 0
        return [x[SINGLE_OBJECT_KEY] for x in self.objects_state_data]

    @property
    def N(self):
        return [len(arr) for arr in self.objects_state_data]
