from typing import List


class GroundTruthConfig:
    def __init__(
        self,
        object_configs: List,
        total_time: int,
        *args,
        **kwargs,
    ) -> None:
        """Construct groundtruth objec

        Parameters
        ----------
        n_births : int
            number of objects hypothesised to exist from timestep 1 to timestep K
        object_configs : List[ObjectConfig]
            object initial state with apperaing and death times
        total_time : int
            total tracking time
        """
        from .object_config import Object

        assert isinstance(total_time, int), "Argument of wrong type!"
        assert all(isinstance(x, Object) for x in object_configs), "Argument of wrong type!"
        self.object_configs = object_configs
        self.total_time = total_time

    @property
    def n_births(self):
        return len(self.object_configs)

    def __repr__(self) -> str:
        return self.__class__.__name__ + (
            f"(n_births={self.n_births}, " f"object_configs={self.object_configs}, " f"total_time={self.total_time}, "
        )
