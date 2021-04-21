import unittest

import torch

from howl.data.dataset import (
    AudioClipExample,
    AudioClipMetadata,
    AudioDataset,
    DatasetType,
)
from howl.data.searcher import WordTranscriptSearcher
from howl.data.tokenize import Vocab
from howl.settings import SETTINGS


class TestDataset(AudioDataset[AudioClipMetadata]):
    """Sample dataset for testing"""

    __test__ = False

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        audio_data = torch.zeros(self.sr)

        metadata1 = AudioClipMetadata(transcription="hello world")
        sample1 = AudioClipExample(metadata=metadata1, audio_data=audio_data, sample_rate=self.sr)

        metadata2 = AudioClipMetadata(transcription="happy new year")
        sample2 = AudioClipExample(metadata=metadata2, audio_data=audio_data, sample_rate=self.sr)

        metadata3 = AudioClipMetadata(transcription="what a beautiful world")
        sample3 = AudioClipExample(metadata=metadata3, audio_data=audio_data, sample_rate=self.sr)

        self.samples = [sample1, sample2, sample3]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx) -> AudioClipExample:
        return self.samples[idx]


class TestAudioDataset(unittest.TestCase):
    def test_compute_statistics(self):
        """test compute statistics
        """
        SETTINGS.training.vocab = ["hello", "world"]
        SETTINGS.training.token_type = "word"
        SETTINGS.inference_engine.inference_sequence = [0, 1]

        vocab = Vocab({"Hello": 0, "World": 1}, oov_token_id=2, oov_word_repr="<OOV>")

        searcher = WordTranscriptSearcher(vocab)

        audio_dataset = TestDataset(metadata_list=[], set_type=DatasetType.TRAINING)
        num_samples = len(audio_dataset)
        total_audio_length = len(audio_dataset)

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
        self.assertEqual(stat.vocab_counts["Hello"], 1)
        self.assertEqual(stat.vocab_counts["World"], 2)


if __name__ == "__main__":
    unittest.main()
