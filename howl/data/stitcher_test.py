import tempfile
import unittest
from pathlib import Path

from howl.data.common.tokenizer import TokenType
from howl.data.common.vocab import Vocab
from howl.data.dataset.dataset import DatasetSplit
from howl.data.stitcher import WordStitcher
from howl.dataset.audio_dataset_constants import AudioDatasetType, SampleType
from howl.dataset.howl_audio_dataset import HowlAudioDataset
from howl.dataset_loader.howl_audio_dataset_loader import HowlAudioDatasetLoader
from howl.settings import SETTINGS
from howl.utils import test_utils


class StitcherTest(test_utils.HowlTest, unittest.TestCase):
    """Test case for Stitcher"""

    def test_word_stitcher(self):
        """test word stitcher"""
        inference_sequence = [0, 1, 2]
        vocab = Vocab({"hey": 0, "fire": 1, "fox": 2}, oov_token_id=3, oov_word_repr="<OOV>")
        wakeword = vocab.wakeword(inference_sequence, separator="_")
        dataset_split = DatasetSplit.TRAINING

        ds_kwargs = dict(
            dataset_split=dataset_split,
            sample_rate=SETTINGS.audio.sample_rate,
            mono=SETTINGS.audio.use_mono,
            labeler=test_utils.frame_labeller(vocab, TokenType.WORD),
        )
        aligned_dataset_path = test_utils.howl_audio_datasets_path() / wakeword / SampleType.POSITIVE.value
        loader = HowlAudioDatasetLoader(AudioDatasetType.ALIGNED, aligned_dataset_path)
        aligned_dataset = loader.load_split(**ds_kwargs)
        stitcher = WordStitcher(vocab=vocab, inference_sequence=inference_sequence, validate_stitched_sample=True)

        with tempfile.TemporaryDirectory() as stitched_dataset_path:
            num_samples = 2
            stitcher.generate_stitched_audio_samples(num_samples, Path(stitched_dataset_path), aligned_dataset)

            for audio_idx in range(num_samples):
                audio_file_name = f"{audio_idx}.wav"
                audio_file_path = Path(stitched_dataset_path) / audio_file_name
                gt_audio_file_path = (
                    test_utils.test_data_path()
                    / "stitcher"
                    / HowlAudioDataset.DIR_STITCHED_TEMPLATE.format(dataset_split=dataset_split)
                    / audio_file_name
                )

                self.assertTrue(self.validate_audio_file(audio_file_path, gt_audio_file_path))
