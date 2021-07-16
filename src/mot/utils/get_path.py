import logging
import os
import shutil


def get_project_dir():
    return os.path.dirname(os.path.abspath(__file__))


def get_output_dir():
    out = os.path.join(get_project_dir(), "out")
    os.makedirs(out, exist_ok=True)
    return out


def get_data_dir():
    return os.path.join(get_project_dir(), "data")


def get_images_dir(current_file=__file__, dir_name=".images"):
    current_dir = os.path.dirname(os.path.abspath(current_file))
    image_dir = os.path.join(current_dir, dir_name)
    os.makedirs(image_dir, exist_ok=True)
    return image_dir


def delete_images_dir(current_file=__file__, dir_name=".images") -> None:
    current_dir = os.path.dirname(os.path.abspath(current_file))
    image_dir = os.path.join(current_dir, dir_name)
    try:
        shutil.rmtree(image_dir)
    except FileNotFoundError:
        logging.error("FileNotFoundError")
