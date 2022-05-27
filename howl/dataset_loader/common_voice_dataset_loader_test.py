import os
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path

import pytest

from howl.data.dataset.dataset import DatasetSplit
from howl.dataset.audio_dataset_constants import AudioDatasetType
from howl.dataset_loader.common_voice_dataset_loader import CommonVoiceDatasetLoader
from howl.utils import filesystem_utils, test_utils


class CommonVoiceDatasetLoaderTest(unittest.TestCase):
    """Test case for CommonVoiceDatasetLoader"""

    @contextmanager
    def _setup_test_env(self):
        """prepare an environment for common voice dataset loader test cases by creating necessary folders"""
        temp_dir = tempfile.TemporaryDirectory()
        dataset_path = Path(temp_dir.name) / "dataset/common-voice"
        filesystem_utils.copytree(test_utils.common_voice_dataset_path(), dataset_path)

        try:
            yield dataset_path
        finally:
            temp_dir.cleanup()

    def test_load_splits(self):
        """Test success case of load_splits"""
        with self._setup_test_env() as dataset_path:
            dataset_loader = CommonVoiceDatasetLoader(dataset_path)
            self.assertEqual(dataset_loader.name, AudioDatasetType.COMMON_VOICE.value)
            self.assertEqual(dataset_loader.dataset_path, dataset_path)

            sample_rate = 1000
            train_ds, dev_ds, test_ds = dataset_loader.load_splits(sample_rate=sample_rate)
            self.assertEqual(len(train_ds.metadata_list), 3)
            self.assertEqual(train_ds.dataset_split, DatasetSplit.TRAINING)
            self.assertEqual(train_ds.sample_rate, sample_rate)

            self.assertEqual(len(dev_ds.metadata_list), 2)
            self.assertEqual(dev_ds.dataset_split, DatasetSplit.DEV)
            self.assertEqual(dev_ds.sample_rate, sample_rate)

            self.assertEqual(len(test_ds.metadata_list), 2)
            self.assertEqual(test_ds.dataset_split, DatasetSplit.TEST)
            self.assertEqual(test_ds.sample_rate, sample_rate)

    def test_missing_dataset(self):
        """Test failure case caused by missing dataset"""
        temp_dir = tempfile.TemporaryDirectory()
        invalid_dataset_path = Path(temp_dir.name) / "empty_dir"

        with pytest.raises(FileNotFoundError):
            CommonVoiceDatasetLoader(invalid_dataset_path)

        temp_dir.cleanup()

    def test_missing_tsv_file(self):
        """Test failure case of load_splits caused by missing tsv file"""
        with self._setup_test_env() as dataset_path:
            # delete test.tsv
            os.remove(dataset_path / "test.tsv")

            dataset_loader = CommonVoiceDatasetLoader(dataset_path)
            self.assertEqual(dataset_loader.name, AudioDatasetType.COMMON_VOICE.value)
            self.assertEqual(dataset_loader.dataset_path, dataset_path)

            with pytest.raises(FileNotFoundError):
                dataset_loader.load_splits()
