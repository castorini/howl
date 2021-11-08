import os
from pathlib import Path


def root_path():
    """Locate root dir of the repo using .root_folder_marker"""
    dir_path = Path(os.path.dirname(os.path.realpath(__file__)))
    while not (dir_path / ".root_folder_marker").exists():
        dir_path = dir_path.parent
    return dir_path


def datasets_path():
    """datasets directory located at the root of the project"""
    return root_path() / "datasets"
