import tempfile
import unittest
from pathlib import Path

from howl.data.dataset_loader.common_voice_dataset_loader import CommonVoiceDatasetLoader
from howl.data.dataset_loader.dataset_loader_factory import DatasetLoaderType, get_dataset_loader


class TestMozillaCommonVoiceLoader(unittest.TestCase):
    def test_missing_dataset(self):
        """Test failure case caused by missing dataset"""
        temp_dir = tempfile.TemporaryDirectory()
        dataset_path = Path(temp_dir.name)

        cv_dataset_loader = get_dataset_loader(DatasetLoaderType.COMMON_VOICE_DATASET_LOADER, dataset_path)
        self.assertIsInstance(cv_dataset_loader, CommonVoiceDatasetLoader)

        temp_dir.cleanup()
