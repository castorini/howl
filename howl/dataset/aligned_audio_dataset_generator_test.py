import os
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path

from howl.data.common.tokenizer import TokenType
from howl.data.dataset.dataset import DatasetSplit
from howl.dataset.aligned_audio_dataset_generator import AlignedAudioDatasetGenerator, AlignmentType
from howl.dataset.audio_dataset_constants import (
    METADATA_FILE_NAME_TEMPLATES,
    METADATA_FILE_PREFIX,
    AudioDatasetType,
    SampleType,
)
from howl.dataset.howl_audio_dataset import HowlAudioDataset
from howl.dataset_loader.howl_audio_dataset_loader import HowlAudioDatasetLoader
from howl.settings import SETTINGS
from howl.utils import filesystem_utils, test_utils


class AlignedAudioDatasetGeneratorTest(test_utils.HowlTest, unittest.TestCase):
    """Test case for AlignedAudioDatasetGenerator"""

    @contextmanager
    def _setup_test_env(self, sample_type: SampleType):
        """prepare raw audio dataset with alignment for aligned audio dataset generator test cases"""
        temp_dir = tempfile.TemporaryDirectory()
        dataset_path = Path(temp_dir.name) / test_utils.WAKEWORD / sample_type.value
        filesystem_utils.copytree(
            test_utils.howl_audio_datasets_path() / test_utils.WAKEWORD / sample_type.value, dataset_path
        )
        alignments_path = dataset_path / HowlAudioDataset.DIR_ALIGNMENT

        aligned_metadata_paths = dataset_path.glob(f"{METADATA_FILE_PREFIX[AudioDatasetType.ALIGNED]}*")
        for aligned_metadata_path in aligned_metadata_paths:
            gt_aligned_metadata_path = aligned_metadata_path.with_name("gt-" + aligned_metadata_path.name)
            os.rename(aligned_metadata_path, gt_aligned_metadata_path)

        try:
            yield dataset_path, alignments_path
        finally:
            temp_dir.cleanup()

    def test_aligned_dataset_generation_for_word_from_mfa_alignment(self):
        """Test aligned metadata generation using mfa alignments"""
        with self._setup_test_env(SampleType.POSITIVE) as (dataset_path, alignments_path):

            SETTINGS.training.vocab = ["The"]
            token_type = TokenType.WORD
            alignment_type = AlignmentType.MFA
            aligned_dataset_generator = AlignedAudioDatasetGenerator(
                dataset_path, alignment_type, alignments_path, token_type=token_type
            )

            self.assertEqual(len(aligned_dataset_generator.train_ds), 2)
            self.assertEqual(len(aligned_dataset_generator.dev_ds), 1)
            self.assertEqual(len(aligned_dataset_generator.test_ds), 0)

            aligned_dataset_generator.generate_datasets()

            for dataset_split in [DatasetSplit.TRAINING, DatasetSplit.DEV, DatasetSplit.TEST]:
                aligned_metadata_path = dataset_path / METADATA_FILE_NAME_TEMPLATES[AudioDatasetType.ALIGNED].format(
                    dataset_split=dataset_split
                )
                gt_aligned_metadata_path = aligned_metadata_path.with_name("gt-" + aligned_metadata_path.name)
                self.assertTrue(test_utils.compare_files(aligned_metadata_path, gt_aligned_metadata_path))

            # make sure the generated datasets are valid
            loader = HowlAudioDatasetLoader(AudioDatasetType.ALIGNED, dataset_path)
            loader.load_splits()

    def test_aligned_dataset_generation_for_word_from_stub_alignment(self):
        """Test aligned metadata generation using stub alignments"""
        with self._setup_test_env(SampleType.NEGATIVE) as (dataset_path, _):

            SETTINGS.training.vocab = ["The"]
            token_type = TokenType.WORD
            alignment_type = AlignmentType.STUB
            aligned_dataset_generator = AlignedAudioDatasetGenerator(
                dataset_path, alignment_type, token_type=token_type
            )

            self.assertEqual(len(aligned_dataset_generator.train_ds), 1)
            self.assertEqual(len(aligned_dataset_generator.dev_ds), 1)
            self.assertEqual(len(aligned_dataset_generator.test_ds), 2)

            aligned_dataset_generator.generate_datasets()

            for dataset_split in [DatasetSplit.TRAINING, DatasetSplit.DEV, DatasetSplit.TEST]:
                aligned_metadata_path = dataset_path / METADATA_FILE_NAME_TEMPLATES[AudioDatasetType.ALIGNED].format(
                    dataset_split=dataset_split
                )
                gt_aligned_metadata_path = aligned_metadata_path.with_name("gt-" + aligned_metadata_path.name)
                self.assertTrue(test_utils.compare_files(aligned_metadata_path, gt_aligned_metadata_path))

            # make sure the generated datasets are valid
            loader = HowlAudioDatasetLoader(AudioDatasetType.ALIGNED, dataset_path)
            loader.load_splits()
