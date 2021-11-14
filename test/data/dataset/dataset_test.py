import unittest

import torch

from howl.data.common.example import AudioClipExample
from howl.data.common.metadata import AudioClipMetadata
from howl.data.common.searcher import WordTranscriptSearcher
from howl.data.common.vocab import Vocab
from howl.data.dataset.dataset import AudioDataset, DatasetType
from howl.settings import SETTINGS
from howl.utils import test_utils
from howl.utils.audio import silent_load


class TestDataset(AudioDataset[AudioClipMetadata]):
    """Sample dataset for testing"""

    __test__ = False

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        common_voice_dataset_path = test_utils.common_voice_dataset_path()
        metadata1 = AudioClipMetadata(
            transcription="The applicants are invited for coffee and visa is given immediately.",
            path=common_voice_dataset_path / "clips/common_voice_en_20005954.mp3",
        )
        audio_data = torch.from_numpy(silent_load(str(metadata1.path), self.sample_rate, self.mono))
        sample1 = AudioClipExample(metadata=metadata1, audio_data=audio_data, sample_rate=self.sample_rate)

        metadata2 = AudioClipMetadata(
            transcription="The anticipated synergies of the two modes of transportation were entirely absent.",
            path=common_voice_dataset_path / "clips/common_voice_en_20009653.mp3",
        )
        audio_data = torch.from_numpy(silent_load(str(metadata2.path), self.sample_rate, self.mono))
        sample2 = AudioClipExample(metadata=metadata2, audio_data=audio_data, sample_rate=self.sample_rate)

        metadata3 = AudioClipMetadata(
            transcription="The fossil fuels include coal, petroleum and natural gas.",
            path=common_voice_dataset_path / "clips/common_voice_en_20009655.mp3",
        )
        audio_data = torch.from_numpy(silent_load(str(metadata3.path), self.sample_rate, self.mono))
        sample3 = AudioClipExample(metadata=metadata3, audio_data=audio_data, sample_rate=self.sample_rate)

        self.samples = [sample1, sample2, sample3]
        self.metadata_list = [metadata1, metadata2, metadata3]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx) -> AudioClipExample:
        return self.samples[idx]


class TestAudioDataset(unittest.TestCase):
    """Sample audio dataset for testing"""

    def test_compute_statistics(self):
        """test compute statistics"""
        SETTINGS.training.vocab = ["The", "and"]
        SETTINGS.training.token_type = "word"
        SETTINGS.inference_engine.inference_sequence = [0, 1]

        vocab = Vocab({"the": 0, "and": 1}, oov_token_id=2, oov_word_repr="<OOV>")

        searcher = WordTranscriptSearcher(vocab)

        audio_dataset = TestDataset(metadata_list=[], set_type=DatasetType.TRAINING)
        num_samples = len(audio_dataset)
        total_audio_length = 15.552

        # without word_searcher, compute_length=True
        stat = audio_dataset.compute_statistics()
        self.assertEqual(stat.num_examples, num_samples)
        self.assertEqual(stat.audio_length_seconds, total_audio_length)
        self.assertFalse(stat.vocab_counts)

        # with word_searcher, compute_length=True
        stat = audio_dataset.compute_statistics(word_searcher=searcher)
        self.assertEqual(stat.num_examples, num_samples)
        self.assertEqual(stat.audio_length_seconds, total_audio_length)
        self.assertEqual(len(stat.vocab_counts), len(vocab))
        self.assertEqual(stat.vocab_counts["the"], 4)
        self.assertEqual(stat.vocab_counts["and"], 2)


if __name__ == "__main__":
    unittest.main()
