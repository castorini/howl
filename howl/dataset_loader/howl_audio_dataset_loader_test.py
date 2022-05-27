import os
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path

import pytest

from howl.data.dataset.dataset import DatasetSplit
from howl.dataset.audio_dataset_constants import METADATA_FILE_NAME_TEMPLATES, AudioDatasetType, SampleType
from howl.dataset_loader.howl_audio_dataset_loader import HowlAudioDatasetLoader
from howl.utils import filesystem_utils, test_utils


class HowlAudioDatasetLoaderTest(unittest.TestCase):
    """Test case for HowlAudioDatasetLoader"""

    @contextmanager
    def _setup_dataset(self):
        """prepare an environment for raw audio dataset loader test cases by creating necessary folders"""
        temp_dir = tempfile.TemporaryDirectory()
        dataset_path = Path(temp_dir.name) / "the" / SampleType.POSITIVE.value
        filesystem_utils.copytree(
            test_utils.howl_audio_datasets_path() / "the" / SampleType.POSITIVE.value, dataset_path
        )

        try:
            yield dataset_path
        finally:
            temp_dir.cleanup()

    def test_missing_dataset(self):
        """Test failure case caused by missing dataset"""
        temp_dir = tempfile.TemporaryDirectory()
        invalid_dataset_path = Path(temp_dir.name) / "empty_dir"

        with pytest.raises(FileNotFoundError):
            HowlAudioDatasetLoader(AudioDatasetType.RAW, invalid_dataset_path)

        with pytest.raises(FileNotFoundError):
            HowlAudioDatasetLoader(AudioDatasetType.ALIGNED, invalid_dataset_path)

        temp_dir.cleanup()

    def test_missing_tsv_file(self):
        """Test failure case of load_splits caused by missing tsv file"""
        with self._setup_dataset() as dataset_path:
            # delete metadata file for raw dataset
            os.remove(
                dataset_path
                / METADATA_FILE_NAME_TEMPLATES[AudioDatasetType.RAW].format(dataset_split=DatasetSplit.TRAINING)
            )

            dataset_loader = HowlAudioDatasetLoader(AudioDatasetType.RAW, dataset_path)
            self.assertEqual(dataset_loader.name, AudioDatasetType.RAW.value)
            self.assertEqual(dataset_loader.dataset_path, dataset_path)

            with pytest.raises(FileNotFoundError):
                dataset_loader.load_splits()

            # delete metadata file for aligned dataset
            os.remove(
                dataset_path
                / METADATA_FILE_NAME_TEMPLATES[AudioDatasetType.ALIGNED].format(dataset_split=DatasetSplit.TRAINING)
            )

            dataset_loader = HowlAudioDatasetLoader(AudioDatasetType.ALIGNED, dataset_path)
            self.assertEqual(dataset_loader.name, AudioDatasetType.ALIGNED.value)
            self.assertEqual(dataset_loader.dataset_path, dataset_path)

            with pytest.raises(FileNotFoundError):
                dataset_loader.load_splits()

    def _test_load_splits(self, dataset_type: AudioDatasetType):
        """Helper function that tests load_splits"""
        with self._setup_dataset() as dataset_path:
            dataset_loader = HowlAudioDatasetLoader(dataset_type, dataset_path)
            self.assertEqual(dataset_loader.name, dataset_type.value)
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

    def test_load_splits(self):
        """Test success case of load_splits"""
        self._test_load_splits(AudioDatasetType.RAW)
        self._test_load_splits(AudioDatasetType.ALIGNED)
