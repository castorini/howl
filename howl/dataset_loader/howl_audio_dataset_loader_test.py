import os
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path

import pytest

from howl.data.dataset.dataset import DatasetSplit
from howl.dataset.audio_dataset_constants import AudioDatasetType
from howl.dataset_loader.howl_audio_dataset_loader import HowlAudioDatasetLoader
from howl.utils import filesystem_utils, test_utils


class HowlAudioDatasetLoaderTest(unittest.TestCase):
    """Test case for HowlAudioDatasetLoader"""

    @contextmanager
    def _setup_test_env(self):
        """prepare an environment for raw audio dataset loader test cases by creating necessary folders"""
        temp_dir = tempfile.TemporaryDirectory()
        dataset_path = Path(temp_dir.name) / "the" / "positive"
        filesystem_utils.copytree(test_utils.raw_audio_datasets_path() / "the" / "positive", dataset_path)

        try:
            yield dataset_path
        finally:
            temp_dir.cleanup()

    def test_load_splits(self):
        """Test success case of load_splits"""
        with self._setup_test_env() as dataset_path:
            dataset_loader = HowlAudioDatasetLoader(AudioDatasetType.RAW, dataset_path)
            self.assertEqual(dataset_loader.name, AudioDatasetType.RAW.value)
            self.assertEqual(dataset_loader.dataset_path, dataset_path)

            sample_rate = 1000
            train_ds, dev_ds, test_ds = dataset_loader.load_splits(sample_rate=sample_rate)
            self.assertEqual(len(train_ds.metadata_list), 2)
            self.assertEqual(train_ds.dataset_split, DatasetSplit.TRAINING)
            self.assertEqual(train_ds.sample_rate, sample_rate)

            self.assertEqual(len(dev_ds.metadata_list), 1)
            self.assertEqual(dev_ds.dataset_split, DatasetSplit.DEV)
            self.assertEqual(dev_ds.sample_rate, sample_rate)

            self.assertEqual(len(test_ds.metadata_list), 0)
            self.assertEqual(test_ds.dataset_split, DatasetSplit.TEST)
            self.assertEqual(test_ds.sample_rate, sample_rate)

    def test_missing_dataset(self):
        """Test failure case caused by missing dataset"""
        temp_dir = tempfile.TemporaryDirectory()
        invalid_dataset_path = Path(temp_dir.name) / "empty_dir"

        with pytest.raises(FileNotFoundError):
            HowlAudioDatasetLoader(AudioDatasetType.RAW, invalid_dataset_path)

        temp_dir.cleanup()

    def test_missing_tsv_file(self):
        """Test failure case of load_splits caused by missing tsv file"""
        with self._setup_test_env() as dataset_path:
            # delete one of the metadata file
            os.remove(dataset_path / "metadata-training.jsonl")

            dataset_loader = HowlAudioDatasetLoader(AudioDatasetType.RAW, dataset_path)
            self.assertEqual(dataset_loader.name, AudioDatasetType.RAW.value)
            self.assertEqual(dataset_loader.dataset_path, dataset_path)

            with pytest.raises(FileNotFoundError):
                dataset_loader.load_splits()
