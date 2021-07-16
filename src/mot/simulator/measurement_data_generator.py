import numpy as np


class MeasurementData:
    """Generates object_generated measurements and clutter"""

    def __init__(
        self,
        object_data,
        sensor_model,
        meas_model,
        random_state=None,
    ):
        """Generates object generated measurement and clutter

        Args:
            object_data (list of lists of Gaussians): a structure contains object data
            sensor_model (SensorModel): a structure specifies sensor model parameters
            meas_model (MeasurementModel): a structure specifies the measurement model
                                           parameters

        Attributes:


        Returns:
            meas_data ([]): cell array of size (total_tracking_time, 1)
                            each cell stores measurements of size
                            (measurement dimension) x (number of measurements
                            at corresponding timestep)
        """
        self._generator = np.random.RandomState(random_state)
        self.object_data = object_data
        self.sensor_model = sensor_model
        self.meas_model = meas_model
        self.meas_data, self.clutter_data = self.generate()

    def generate(self):
        meas_data = [[] for timestep in range(len(self.object_data))]
        clutter_data = [[] for timestep in range(len(self.object_data))]

        for timestep in range(len(self.object_data)):
            number_of_objects_in_scene = self.object_data.N[timestep]
            if number_of_objects_in_scene > 0:
                detection_mask = self._generator.uniform(size=number_of_objects_in_scene) < self.sensor_model.P_D
                observed_objects = [
                    self.object_data[timestep][key]
                    for is_observed, key in zip(detection_mask, self.object_data[timestep].keys())
                    if is_observed
                ]
                # Generate measurement
                for observed_object in observed_objects:
                    measurement = self.meas_model.observe(observed_object.x)
                    meas_data[timestep].append(measurement)
            # Number if clutter measurements
            N_c = self._generator.poisson(lam=self.sensor_model.lambda_c)
            # Generate clutter
            # TODO keep in mind that pos of object can be vary
            clutter_min_coord, clutter_max_coord = np.diag(self.sensor_model.range_c)
            clutter = np.random.uniform(clutter_min_coord, clutter_max_coord, [N_c, 2])
            clutter_data[timestep].append(clutter)
        return tuple(meas_data), tuple(clutter_data)

    def __len__(self):
        return len(self.object_data)

    def __getitem__(self, key):
        return np.array(self._observed_measurements[key])

    def __iter__(self):
        self.n = 0
        return self

    def __next__(self):
        if self.n < self.__len__():
            result = np.array(self._observed_measurements[self.n])
            self.n += 1
            return result
        else:
            raise StopIteration

    def __repr__(self) -> str:
        return self.__class__.__name__ + (
            f"(sensor_model={self.sensor_model}, "
            f"meas_model={self.meas_model}, "
            f"X={self._observed_measurements}, "
        )

    @property
    def _observed_measurements(self):
        measurements_with_clutter = []
        for _idx, (curr_measurements, curr_clutters) in enumerate(zip(self.meas_data, self.clutter_data)):
            scene = []
            for cur_measurement in curr_measurements:
                scene.append(cur_measurement)
            for curr_clutter in curr_clutters[0]:
                scene.append(curr_clutter)
            measurements_with_clutter.append(scene)
        return measurements_with_clutter
