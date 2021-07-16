import ast
import json
import logging
from collections import UserDict
from dataclasses import dataclass
from typing import Dict

import numpy as np
import pytest
import tqdm
import ujson
from nuscenes.eval.common.data_classes import EvalBoxes
from nuscenes.eval.common.loaders import load_gt
from nuscenes.eval.detection.data_classes import DetectionBox
from nuscenes.eval.tracking.data_classes import TrackingBox
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.splits import (  # noqa: F401
    mini_train,
    mini_val,
    test,
    train,
    train_detect,
    train_track,
    val,
)
from pyquaternion import Quaternion

import mot.motion_models
from mot.common import Gaussian, GaussianMixture, WeightedGaussian
from mot.common.state import ObjectMetadata, Observation, ObservationList
from mot.configs import SensorModelConfig
from mot.evaluation_runners.evaluator import OneSceneMOTevaluator
from mot.trackers.multiple_object_trackers.PMBM.common.birth_model import (
    MeasurementDrivenBirthModel,
)
from mot.trackers.multiple_object_trackers.PMBM.pmbm import PMBM


# create logger
logger = logging.getLogger("nuscenes_eval")
logger.setLevel(logging.DEBUG)


@dataclass
class NuscenesDatasetConfig:
    data_path: str  # path to data
    version: str  # datasetVersion


# @dataclass
# class ObjectGroundTruth:
#     center_translation:
#     center_roatation
#     velocity:
#     object_class:
#     size:


class NuscenceInstanceTokenReidentification(UserDict):
    def __init__(self):
        self.max_id = 0
        self.data = {}

    def add_id(self, instance_token):
        if instance_token not in self.data.keys():
            self.data[instance_token] = self.max_id
            self.max_id += 1


class NuscenesTrackerEvaluator:
    def __init__(self, detection_filepath: str, nuscens_config: NuscenesDatasetConfig, evaluation_set: str) -> None:
        # with mp.Pool(processes=8) as pool:

        #     # result1 = pool.map_async(self.initialize_nuscenes, [nuscens_config])

        #     # result2 = pool.map_async(self.read_detection_file, [detection_filepath])
        #     # result1 = result1.get()
        #     # result2 = result2.get()
        _, self.detection_metadata, self.detection_results = self.read_detection_file(detection_filepath)
        self.nuscenes_helper = self.initialize_nuscenes(nuscens_config)
        self.evaluation_set = tuple(ast.literal_eval(evaluation_set))

        # check if dataset contains all scenes from detection file
        assert set(self.evaluation_set).issubset(set([scene["name"] for scene in self.nuscenes_helper.scene]))

        self.results_by_scene = {}

    def run_tracking_evaluation(self):
        """Returns data in nuscenes tracking submission format.
        submission {
            "meta": {
                "use_camera":   <bool>  -- Whether this submission uses camera data as an input.
                "use_lidar":    <bool>  -- Whether this submission uses lidar data as an input.
                "use_radar":    <bool>  -- Whether this submission uses radar data as an input.
                "use_map":      <bool>  -- Whether this submission uses map data as an input.
                "use_external": <bool>  -- Whether this submission uses external data as an input.
            },
            "results": {
                sample_token <str>: List[sample_result] -- Maps each sample_token to a list of sample_results.
            }
        }
        """
        results = {}

        for scene_name in self.evaluation_set:
            print(f"scene name: {scene_name}")
            scene = [scene for scene in self.nuscenes_helper.scene if scene["name"] == scene_name][0]
            scene_token = scene["token"]
            results.update(self.process_scene(scene_token))

        gt_boxes = load_gt(self.nuscenes_helper, "mini_val", TrackingBox, verbose=True)

        for sample_token in gt_boxes.sample_tokens:
            if sample_token in results.keys():
                continue
            else:
                results[sample_token] = []

        output_path = "results.json"
        output_data = {"meta": self.detection_metadata, "results": results}
        with open(output_path, "w") as outfile:
            json.dump(output_data, outfile)

    def process_scene(self, scene_token: str) -> Dict:

        scene_object = self.nuscenes_helper.get(table_name="scene", token=scene_token)
        logging.debug(f"Process scene with name {scene_object['name']}")
        first_sample_token = scene_object["first_sample_token"]
        scene_sample_tokens_ = self.get_scene_token_list(firts_frame_token=first_sample_token)
        scene_sample_tokens = tuple([first_sample_token] + list(scene_sample_tokens_))
        detection_probability = 0.7
        dt = 0.5
        birth_model = GaussianMixture(
            [
                WeightedGaussian(
                    np.log(0.03),
                    Gaussian(x=np.array([620.0, 1640.0, 0.0, 0.0]), P=100 * np.eye(4)),
                ),
                WeightedGaussian(
                    np.log(0.03),
                    Gaussian(x=np.array([680.0, 1680.0, 0.0, 0.0]), P=100 * np.eye(4)),
                ),
                WeightedGaussian(
                    np.log(0.03),
                    Gaussian(x=np.array([580.0, 1600.0, 0.0, 0.0]), P=100 * np.eye(4)),
                ),
                WeightedGaussian(
                    np.log(0.03),
                    Gaussian(x=np.array([680.0, 1600.0, 0.0, 0.0]), P=100 * np.eye(4)),
                ),
            ]
        )

        tracker = PMBM(
            meas_model=mot.measurement_models.ConstantVelocityMeasurementModel(sigma_r=0.5),
            sensor_model=SensorModelConfig(
                P_D=detection_probability,
                lambda_c=10.0,
                range_c=np.array([[500, 1000], [1500, 1700]]),
            ),
            motion_model=mot.motion_models.ConstantVelocityMotionModel(dt, sigma_q=5.0),
            birth_model=MeasurementDrivenBirthModel(),
            max_number_of_hypotheses=30,
            gating_percentage=1.0,
            detection_probability=detection_probability,
            survival_probability=0.9,
            existense_probability_threshold=0.6,
            track_history_length_threshold=2,
            density=mot.common.GaussianDensity,
            initial_PPP_intensity=birth_model,
        )
        scene_estimations = []
        result_estimations = {}
        scene_measurements = []
        result_results_scene = {}

        gt = []
        id_pool = NuscenceInstanceTokenReidentification()

        logging.info(f"Tracking {scene_token} scene")
        evaluator = OneSceneMOTevaluator()

        for timestep, sample_token in tqdm.tqdm(enumerate(scene_sample_tokens)):
            measurements = self.get_measurements_for_one_sample(token=sample_token)
            logging.info(f"tracker condition: {tracker}")
            logging.info(f"got {len(measurements)} measurements")

            current_sample_data = self.nuscenes_helper.get("sample", sample_token)
            lidar_top_data_token = current_sample_data["data"]["LIDAR_TOP"]
            lidar_data = self.nuscenes_helper.get("sample_data", lidar_top_data_token)
            lidar_top_ego_pose = self.nuscenes_helper.get("ego_pose", lidar_data["ego_pose_token"])
            current_ego_pose = lidar_top_ego_pose

            estimation = tracker.step(measurements, dt, current_ego_pose)

            scene_measurements.append(measurements)
            scene_estimations.append(estimation)

            annotations = [
                self.nuscenes_helper.get("sample_annotation", annotation_token)
                for annotation_token in current_sample_data["anns"]
            ]

            scene_data = {}
            for annotation in annotations:
                if annotation["category_name"].split(".")[0] == "vehicle" and annotation["num_lidar_pts"] > 10:
                    id_pool.add_id(annotation["instance_token"])
                    object_id = id_pool.data[annotation["instance_token"]]
                    object_pos_x, object_pos_y = annotation["translation"][:2]
                    scene_data[object_id] = Gaussian(x=np.array([object_pos_x, object_pos_y, 0.0, 0.0]), P=np.eye(4))

            evaluator.step(
                sample_measurements=measurements, sample_estimates=estimation, sample_gt=scene_data, timestep=timestep
            )

            gt.append(annotations)
            if not annotations:
                logging.info("annotation empty")

            def format_nuscenes_sample_results(sample_token: str, tracker_estimations):

                if tracker_estimations:
                    results = []
                    for tracker_estimation_kv in tracker_estimations:
                        object_id = list(tracker_estimation_kv.keys())[0]
                        tracker_estimation = tracker_estimation_kv[object_id]
                        translation = [list(tracker_estimation)[0], list(tracker_estimation)[1], 0]  # x, y, z
                        size = [1.0, 1.0, 1.0]  # w, h ,l
                        rotation = Quaternion(axis=[0, 0, 1], angle=0).elements
                        velocity = [list(tracker_estimation)[0], list(tracker_estimation)[1]]  # vx, vy
                        sample_result = {
                            "sample_token": sample_token,
                            "translation": translation,
                            "size": size,
                            "rotation": list(rotation),
                            "velocity": velocity,
                            "tracking_id": object_id,
                            "tracking_name": "car",
                            "tracking_score": 0.8,
                        }
                        results.append(sample_result)

                    return {sample_token: results}

            if estimation:
                nuscenes_results = format_nuscenes_sample_results(sample_token, estimation)
                result_estimations[sample_token] = nuscenes_results
                result_results_scene.update(nuscenes_results)

        evaluator.post_processing()
        # meta = f"scene_token={scene_token}"
        # plt.savefig(get_images_dir(__file__) + "/" + "results_" + meta + ".png")
        return result_results_scene

    def get_measurements_for_one_sample(self, token):
        detection_results = self.detection_results[token]
        measurements = []

        for detection in detection_results:
            obj_type = detection.detection_name

            # get object pose
            pos_x, pos_y, pos_z = detection.translation
            quaternion = Quaternion(detection.rotation)
            yaw = quaternion.angle if quaternion.axis[2] > 0 else -quaternion.angle
            print(yaw)

            object_size = detection.size
            detection_confidence = detection.detection_score

            # TODO
            object_detection = Observation(
                measurement=np.array([pos_x, pos_y]),
                metadata=ObjectMetadata(object_class=obj_type, size=object_size, confidence=detection_confidence),
            )
            if detection.detection_name == "car" and detection.detection_score > 0.3:
                print(detection.detection_name, detection.detection_score)
                measurements.append(object_detection)  # get only x,y position

        observations = ObservationList(measurements)
        return observations

    def get_scenes_from_detections(self):
        # get list of scenes from detection file
        scene_tokens = set()

        for sample_token in self.detection_results.sample_tokens:
            scene_token = self.nuscenes_helper.get(table_name="sample", token=sample_token)["scene_token"]
            scene_tokens.add(scene_token)
        logging.info(f"Tokens: {scene_tokens}")
        # TODO убрать костыль
        return scene_tokens

    def read_detection_file(self, detection_filepath: str):
        logging.info(f"Reading detection file {detection_filepath}")
        with open(detection_filepath) as file:
            detection_data = ujson.load(file)
        metadata = detection_data["meta"]
        results = EvalBoxes.deserialize(detection_data["results"], DetectionBox)
        return detection_data, metadata, results

    def initialize_nuscenes(self, config: NuscenesDatasetConfig):
        nuscenes = NuScenes(version=config.version, dataroot=config.data_path, verbose=True)
        return nuscenes

    def get_scene_token_list(self, firts_frame_token: str):
        samples = self.traverse_linked_list(
            nuscenes=self.nuscenes_helper,
            first_object=self.nuscenes_helper.get("sample", firts_frame_token),
            direction="next",
        )
        frame_tokens = [sample["token"] for sample in samples]
        return tuple(frame_tokens)

    def traverse_linked_list(
        self,
        nuscenes: NuScenes,
        first_object,
        table_hey: str = "sample",
        direction: str = "prev",
        inclusive: bool = False,
    ):
        assert direction in ("prev", "next")

        current_object = first_object

        if not inclusive:
            sequence = []
        else:
            sequence = [current_object]

        while True:
            if len(current_object[direction]):
                current_object = nuscenes.get(table_name=table_hey, token=current_object[direction])
                sequence.append(current_object)
            else:
                break
        if direction == "prev":
            return sequence[::-1]
        else:
            return sequence


@pytest.fixture
def nuscenes_config():
    yield NuscenesDatasetConfig(
        data_path="./data/nuscenes/dataset/v1.0-mini",  # noqa
        version="v1.0-mini",
    )


def test_nuscenes(nuscenes_config):
    evaluator = NuscenesTrackerEvaluator(
        detection_filepath="./data/nuscenes/detection-megvii/megvii_val.json",  # noqa
        nuscens_config=nuscenes_config,
    )
    evaluator.evaluate()


if __name__ == "__main__":
    test_nuscenes()
