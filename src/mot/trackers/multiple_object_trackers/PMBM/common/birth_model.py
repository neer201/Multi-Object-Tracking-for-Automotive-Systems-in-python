from copy import deepcopy

import numpy as np

from mot.common import Gaussian, GaussianMixture, WeightedGaussian


class BirthModel:
    def get_born_objects_intensity(self, params) -> GaussianMixture:
        raise NotImplementedError


class StaticBirthModel(BirthModel):
    def __init__(self, birth_model_config: GaussianMixture):
        self.birth_model_config = deepcopy(birth_model_config)
        super(StaticBirthModel, self).__init__()

    def get_born_objects_intensity(self, params=None) -> GaussianMixture:
        birth_model = deepcopy(self.birth_model_config)

        if params is not None:
            ego_pose = params["ego_pose"]
            translation, rotation = ego_pose["translation"], ego_pose["rotation"]  # noqa

            for i, _birth_component in enumerate(birth_model):
                birth_model[i].gaussian.x[:2] += translation[:2]

        return birth_model


class RandomSampledBirthModel(BirthModel):
    def __init__(self):
        super(RandomSampledBirthModel, self).__init__()
        raise NotImplementedError


class MeasurementDrivenBirthModel(BirthModel):
    def __init__(self):
        super(MeasurementDrivenBirthModel, self).__init__()

    def get_born_objects_intensity(self, params) -> GaussianMixture:
        measurements = params["measurements"]
        num_of_born_components_per_measurement = 10

        generated_intensity = []

        random_width = 10
        for measurement in measurements:
            delta_x = np.random.uniform(-random_width, random_width, num_of_born_components_per_measurement)
            delta_y = np.random.uniform(-random_width, random_width, num_of_born_components_per_measurement)

            for d_x, d_y in zip(delta_x, delta_y):
                state = np.array([measurement.measurement[0] + d_x, measurement.measurement[1] + d_y, 0.0, 0.0])
                sample_gaussian = Gaussian(x=state, P=100 * np.eye(4))
                generated_intensity.append(WeightedGaussian(log_weight=np.log(0.03), gaussian=sample_gaussian))
        return GaussianMixture(generated_intensity)
