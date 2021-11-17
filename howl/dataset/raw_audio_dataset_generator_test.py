import glob
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path

from howl.dataset.raw_audio_dataset_generator import RawAudioDatasetGenerator, SampleType
from howl.dataset_loader.dataset_loader_factory import DatasetLoaderType
from howl.settings import SETTINGS
from howl.utils import filesystem_utils, test_utils


class RawAudioDatasetGeneratorTest(unittest.TestCase):
    """Test case for RawAudioDatasetGenerator"""

    @contextmanager
    def _setup_test_env(self):
        """prepare an environment for ml-pipeline test cases by creating necessary folders"""
        temp_dir = tempfile.TemporaryDirectory()
        temp_dir_path = Path(temp_dir.name)
        dataset_path = temp_dir_path / "dataset/common-voice"
        filesystem_utils.copytree(test_utils.common_voice_dataset_path(), dataset_path)

        try:
            yield temp_dir_path, dataset_path
        finally:
            temp_dir.cleanup()

    def test_generate_datasets(self):
        """Test generate datasets"""
        with self._setup_test_env() as (temp_dir_path, input_dataset_path):
            dataset_loader_type = DatasetLoaderType.COMMON_VOICE_DATASET_LOADER

            SETTINGS.training.vocab = ["The"]
            SETTINGS.training.token_type = "word"

            raw_dataset_generator = RawAudioDatasetGenerator(input_dataset_path, dataset_loader_type,)

            self.assertEqual(raw_dataset_generator.input_dataset_path, input_dataset_path)
            self.assertEqual(raw_dataset_generator.dataset_loader_type, dataset_loader_type)
            self.assertEqual(raw_dataset_generator.inference_ctx.token_type, SETTINGS.training.token_type)

            positive_dataset = temp_dir_path / "positive"
            raw_dataset_generator.generate_datasets(positive_dataset, SampleType.POSITIVE)
            lab_file_paths = glob.glob(str(positive_dataset / "audio/*.lab"))
            self.assertGreater(len(lab_file_paths), 0)
            for path in lab_file_paths:
                with open(path, "r") as lab_file:
                    self.assertIn(SETTINGS.training.vocab[0], lab_file.readline())

            self.assertEqual(test_utils.get_num_of_lines(positive_dataset / "metadata-training.jsonl"), 2)
            self.assertEqual(test_utils.get_num_of_lines(positive_dataset / "metadata-dev.jsonl"), 1)
            self.assertEqual(test_utils.get_num_of_lines(positive_dataset / "metadata-test.jsonl"), 0)

            negative_dataset = temp_dir_path / "negative"
            raw_dataset_generator.generate_datasets(negative_dataset, SampleType.NEGATIVE)
            lab_file_paths = glob.glob(str(negative_dataset / "audio/*.lab"))
            self.assertGreater(len(lab_file_paths), 0)
            for path in lab_file_paths:
                with open(path, "r") as lab_file:
                    self.assertNotIn(SETTINGS.training.vocab[0], lab_file.readline())

            self.assertEqual(test_utils.get_num_of_lines(negative_dataset / "metadata-training.jsonl"), 1)
            self.assertEqual(test_utils.get_num_of_lines(negative_dataset / "metadata-dev.jsonl"), 1)
            self.assertEqual(test_utils.get_num_of_lines(negative_dataset / "metadata-test.jsonl"), 2)
