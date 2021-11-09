from pathlib import Path

import howl


def test_data_path():
    """Test data folder path"""
    return howl.root_path() / "test/test_data"


def common_voice_dataset_path():
    """Test common voice dataset folder path"""
    return test_data_path() / "dataset/common-voice"


def get_num_of_lines(file_path: Path):
    """Get number of lines in the given file path"""
    return sum(1 for _ in open(file_path))
