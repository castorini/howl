import os
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path

from howl.data.common.tokenizer import TokenType
from howl.dataset.aligned_audio_dataset_generator import AlignedAudioDatasetGenerator, AlignmentType
from howl.settings import SETTINGS
from howl.utils import filesystem_utils, test_utils


class AlignedAudioDatasetGeneratorTest(unittest.TestCase):
    """Test case for AlignedAudioDatasetGenerator"""

    @contextmanager
    def _setup_test_env(self, dataset_type: str):
        """prepare raw audio dataset with alignment for aligned audio dataset generator test cases"""
        temp_dir = tempfile.TemporaryDirectory()
        dataset_path = Path(temp_dir.name) / "the" / dataset_type
        filesystem_utils.copytree(test_utils.howl_audio_datasets_path() / "the" / dataset_type, dataset_path)
        alignments_path = dataset_path / "alignment"

        aligned_metadata_paths = dataset_path.glob(f"{AlignedAudioDatasetGenerator.ALIGNED_METADATA_PREFIX}*")
        for aligned_metadata_path in aligned_metadata_paths:
            gt_aligned_metadata_path = aligned_metadata_path.with_name("gt-" + aligned_metadata_path.name)
            os.rename(aligned_metadata_path, gt_aligned_metadata_path)

        try:
            yield dataset_path, alignments_path
        finally:
            temp_dir.cleanup()

    def test_aligned_dataset_generation_for_word_from_mfa_alignment(self):
        """Test aligned metadata generation using mfa alignments"""
        with self._setup_test_env("positive") as (dataset_path, alignments_path):

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

            for dataset_type in ["training", "dev", "test"]:
                aligned_metadata_path = dataset_path / f"aligned-metadata-{dataset_type}.jsonl"
                gt_aligned_metadata_path = aligned_metadata_path.with_name("gt-" + aligned_metadata_path.name)
                self.assertTrue(test_utils.compare_files(aligned_metadata_path, gt_aligned_metadata_path))

    def test_aligned_dataset_generation_for_word_from_stub_alignment(self):
        """Test aligned metadata generation using stub alignments"""
        with self._setup_test_env("negative") as (dataset_path, _):

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

            for dataset_type in ["training", "dev", "test"]:
                aligned_metadata_path = dataset_path / f"aligned-metadata-{dataset_type}.jsonl"
                gt_aligned_metadata_path = aligned_metadata_path.with_name("gt-" + aligned_metadata_path.name)
                self.assertTrue(test_utils.compare_files(aligned_metadata_path, gt_aligned_metadata_path))
