import os
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path

from howl.data.common.tokenizer import TokenType
from howl.data.common.vocab import Vocab
from howl.data.dataset.dataset import DatasetSplit
from howl.dataset.audio_dataset_constants import (
    METADATA_FILE_NAME_TEMPLATES,
    METADATA_FILE_PREFIX,
    AudioDatasetType,
    SampleType,
)
from howl.dataset.stitched_audio_dataset_generator import StitchedAudioDatasetGenerator
from howl.dataset_loader.howl_audio_dataset_loader import HowlAudioDatasetLoader
from howl.utils import filesystem_utils, test_utils


class StitchedAudioDatasetGeneratorTest(test_utils.HowlTest, unittest.TestCase):
    """Test case for StitchedAudioDatasetGenerator"""

    def _setup(self):
        self.inference_sequence = [0, 1, 2]
        self.vocab = Vocab({"hey": 0, "fire": 1, "fox": 2}, oov_token_id=3, oov_word_repr="<OOV>")

    @contextmanager
    def _setup_test_env(self, sample_type: SampleType):
        """prepare aligned audio dataset with alignment for stitched audio dataset generator test cases"""
        wakeword = self.vocab.wakeword(self.inference_sequence, separator="_")

        temp_dir = tempfile.TemporaryDirectory()
        dataset_path = Path(temp_dir.name) / wakeword / sample_type.value
        filesystem_utils.copytree(test_utils.howl_audio_datasets_path() / wakeword / sample_type.value, dataset_path)

        stitched_metadata_paths = dataset_path.glob(f"{METADATA_FILE_PREFIX[AudioDatasetType.STITCHED]}*")
        for stitched_metadata_path in stitched_metadata_paths:
            gt_stitched_metadata_path = stitched_metadata_path.with_name("gt-" + stitched_metadata_path.name)
            os.rename(stitched_metadata_path, gt_stitched_metadata_path)

        try:
            yield dataset_path
        finally:
            temp_dir.cleanup()

    def test_stitched_dataset_generation_for_word_from_mfa_alignment(self):
        """Test stitched dataset generation"""

        with self._setup_test_env(SampleType.POSITIVE) as (dataset_path):

            max_num_training_samples = 2
            max_num_dev_samples = 1
            max_num_test_samples = 1

            stitched_audio_dataset_generator = StitchedAudioDatasetGenerator(
                aligned_audio_dataset_path=dataset_path,
                vocab=self.vocab,
                max_num_training_samples=max_num_training_samples,
                max_num_dev_samples=max_num_dev_samples,
                max_num_test_samples=max_num_test_samples,
                validate_stitched_sample=True,
                labeller=test_utils.frame_labeller(self.vocab, TokenType.WORD),
            )

            stitched_audio_dataset_generator.generate_datasets()

            for dataset_split in [DatasetSplit.TRAINING, DatasetSplit.DEV, DatasetSplit.TEST]:
                stitched_metadata_path = dataset_path / METADATA_FILE_NAME_TEMPLATES[AudioDatasetType.STITCHED].format(
                    dataset_split=dataset_split
                )
                gt_stitched_metadata_path = stitched_metadata_path.with_name("gt-" + stitched_metadata_path.name)
                self.assertTrue(test_utils.compare_files(stitched_metadata_path, gt_stitched_metadata_path))

            # make sure the generated datasets are valid
            loader = HowlAudioDatasetLoader(AudioDatasetType.STITCHED, dataset_path)
            training_set, dev_set, test_set = loader.load_splits()

            self.assertEqual(len(training_set), max_num_training_samples)
            self.assertEqual(len(dev_set), max_num_dev_samples)
            self.assertEqual(len(test_set), max_num_test_samples)
