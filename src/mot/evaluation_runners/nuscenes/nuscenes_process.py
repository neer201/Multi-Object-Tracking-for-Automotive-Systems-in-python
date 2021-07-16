import argparse
import logging

import yaml

from mot.evaluation_runners.nuscenes.nuscenes_eval import (
    NuscenesDatasetConfig,
    NuscenesTrackerEvaluator,
)


def create_parser():
    parser = argparse.ArgumentParser()

    arg_group = parser.add_argument_group("Targets")
    arg_group.add_argument(
        "--config-file",
        dest="config_file",
        help="configuration file *.yml",
        type=argparse.FileType(mode="r"),
        default="config.yml",
    )

    arg_group.add_argument(
        "--nuscenes-split-set",
        dest="nuscenes_split_set",
        help="nuscenes split set \
            https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/nuscenes/utils/splits.py",
    )
    arg_group.add_argument(
        "--detection-file", dest="detection_filepath", help="detections saved in nuscenes submission format"
    )
    arg_group.add_argument(
        "--nuscenes-version",
        dest="nuscenes_version",
        help="sample or full dataset",
        default="v1.0-mini",
    )

    arg_group.add_argument(
        "-log",
        "--log",
        default="warning",
        help=("Provide logging level. " "Example --log debug', default='warning'"),
    )

    return parser


def parse_args(argument_parser):
    args = argument_parser.parse_args()

    if args.config_file:
        config_data = yaml.load(args.config_file, Loader=yaml.SafeLoader)
        delattr(args, "config_file")
        args_dict = args.__dict__

        for key, value in config_data.items():
            if isinstance(value, list):
                args_dict[key].extend(value)
            else:
                args_dict[key] = value
    return args


if __name__ == "__main__":
    parser = create_parser()
    args = parse_args(parser)

    levels = {
        "critical": logging.CRITICAL,
        "error": logging.ERROR,
        "warn": logging.WARNING,
        "warning": logging.WARNING,
        "info": logging.INFO,
        "debug": logging.DEBUG,
    }
    level = levels.get(args.log.lower())
    if level is None:
        raise ValueError(f"log level given: {args.log}" f" -- must be one of: {' | '.join(levels.keys())}")
    logging.basicConfig(level=level)
    logger = logging.getLogger(__name__)

    nuscenes_tracker_evaluator = NuscenesTrackerEvaluator(
        detection_filepath=args.detection_filepath,
        nuscens_config=NuscenesDatasetConfig(
            data_path="./data/nuscenes/dataset",
            version=args.nuscenes_version,
        ),
        evaluation_set=args.nuscenes_split_set,
    )
    nuscenes_tracker_evaluator.run_tracking_evaluation()
