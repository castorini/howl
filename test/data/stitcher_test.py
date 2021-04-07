import random
import unittest
from pathlib import Path

from howl.data.dataset import WakeWordDatasetLoader, WordFrameLabeler
from howl.data.stitcher import WordStitcher
from howl.data.tokenize import Vocab
from howl.settings import SETTINGS


class TestStitcher(unittest.TestCase):
    def test_compute_statistics(self):
        random.seed(1)

        """test compute statistics
        """
        SETTINGS.training.vocab = ["hey", "fire", "fox"]
        SETTINGS.training.token_type = "word"
        SETTINGS.inference_engine.inference_sequence = [0, 1, 2]

        vocab = Vocab({"hey": 0, "fire": 1, "fox": 2}, oov_token_id=3, oov_word_repr="<OOV>")
        labeler = WordFrameLabeler(vocab)

        loader = WakeWordDatasetLoader()
        ds_kwargs = dict(sr=SETTINGS.audio.sample_rate, mono=SETTINGS.audio.use_mono, frame_labeler=labeler)

        test_dataset_path = Path("test/test_data/stitcher")
        stitched_dataset_path = test_dataset_path / "stitched"
        stitched_dataset_path.mkdir(exist_ok=True)

        test_ds, _, _ = loader.load_splits(test_dataset_path, **ds_kwargs)
        stitcher = WordStitcher(vocab=vocab, detect_keyword=True)
        stitcher.stitch(20, stitched_dataset_path, test_ds)

        stitched_train_ds, stitched_dev_ds, stitched_test_ds = stitcher.load_splits(0.5, 0.25, 0.25)

        self.assertEqual(len(stitched_train_ds), 10)
        self.assertEqual(len(stitched_dev_ds), 5)
        self.assertEqual(len(stitched_test_ds), 5)


if __name__ == "__main__":
    unittest.main()
