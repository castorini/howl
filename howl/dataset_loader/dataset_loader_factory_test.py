import tempfile
import unittest
from pathlib import Path

from howl.dataset_loader.aligned_audio_dataset_loader import AlignedAudioDatasetLoader
from howl.dataset_loader.common_voice_dataset_loader import CommonVoiceDatasetLoader
from howl.dataset_loader.dataset_loader_factory import DatasetLoaderType, get_dataset_loader
from howl.dataset_loader.raw_audio_dataset_loader import RawAudioDatasetLoader


class DatasetLoaderFactoryTest(unittest.TestCase):
    """Test case for dataset_loader_factory"""

    def test_missing_dataset(self):
        """Test failure case caused by missing dataset"""

        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_path = Path(temp_dir)

            cv_dataset_loader = get_dataset_loader(DatasetLoaderType.COMMON_VOICE_DATASET_LOADER, dataset_path)
            self.assertIsInstance(cv_dataset_loader, CommonVoiceDatasetLoader)

            cv_dataset_loader = get_dataset_loader(DatasetLoaderType.RAW_AUDIO_DATASET_LOADER, dataset_path)
            self.assertIsInstance(cv_dataset_loader, RawAudioDatasetLoader)

            cv_dataset_loader = get_dataset_loader(DatasetLoaderType.ALIGNED_AUDIO_DATASET_LOADER, dataset_path)
            self.assertIsInstance(cv_dataset_loader, AlignedAudioDatasetLoader)
