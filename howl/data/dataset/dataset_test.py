import unittest

from howl.data.common.searcher import WordTranscriptSearcher
from howl.data.common.vocab import Vocab
from howl.data.dataset.dataset import DatasetType
from howl.settings import SETTINGS
from howl.utils import test_utils


class TestAudioDataset(unittest.TestCase):
    """Sample audio dataset for testing"""

    def test_compute_statistics(self):
        """test compute statistics"""
        SETTINGS.training.vocab = ["The", "and"]
        SETTINGS.training.token_type = "word"
        SETTINGS.inference_engine.inference_sequence = [0, 1]

        vocab = Vocab({"the": 0, "and": 1}, oov_token_id=2, oov_word_repr="<OOV>")

        searcher = WordTranscriptSearcher(vocab)

        audio_dataset = test_utils.TestDataset(metadata_list=[], set_type=DatasetType.TRAINING)
        num_samples = len(audio_dataset)
        total_audio_length = 15.552

        # without word_searcher, compute_length=True
        stat = audio_dataset.compute_statistics()
        self.assertEqual(stat.num_examples, num_samples)
        self.assertAlmostEqual(stat.audio_length_seconds, total_audio_length, 1)
        self.assertFalse(stat.vocab_counts)

        # with word_searcher, compute_length=True
        stat = audio_dataset.compute_statistics(word_searcher=searcher)
        self.assertEqual(stat.num_examples, num_samples)
        self.assertAlmostEqual(stat.audio_length_seconds, total_audio_length, 1)
        self.assertEqual(len(stat.vocab_counts), len(vocab))
        self.assertEqual(stat.vocab_counts["the"], 4)
        self.assertEqual(stat.vocab_counts["and"], 2)


if __name__ == "__main__":
    unittest.main()
