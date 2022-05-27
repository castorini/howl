import tempfile
import unittest
from pathlib import Path

from howl.dataset.audio_dataset_constants import AudioDatasetType
from howl.dataset_loader.common_voice_dataset_loader import CommonVoiceDatasetLoader
from howl.dataset_loader.dataset_loader_factory import get_dataset_loader
from howl.dataset_loader.howl_audio_dataset_loader import HowlAudioDatasetLoader


class DatasetLoaderFactoryTest(unittest.TestCase):
    """Test case for dataset_loader_factory"""

    def test_missing_dataset(self):
        """Test failure case caused by missing dataset"""

        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_path = Path(temp_dir)

            cv_dataset_loader = get_dataset_loader(AudioDatasetType.COMMON_VOICE, dataset_path)
            self.assertIsInstance(cv_dataset_loader, CommonVoiceDatasetLoader)

            cv_dataset_loader = get_dataset_loader(AudioDatasetType.RAW, dataset_path)
            self.assertIsInstance(cv_dataset_loader, HowlAudioDatasetLoader)

            cv_dataset_loader = get_dataset_loader(AudioDatasetType.ALIGNED, dataset_path)
            self.assertIsInstance(cv_dataset_loader, HowlAudioDatasetLoader)
