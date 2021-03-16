
import unittest
from pathlib import Path

import torch
from howl.data.dataset import (AudioClipExample, AudioClipMetadata,
                               DatasetType, WakeWordDatasetLoader,
                               WordFrameLabeler)
from howl.data.searcher import WordTranscriptSearcher
from howl.data.stitcher import WordStitcher
from howl.data.tokenize import Vocab
from howl.settings import SETTINGS


class TestStitcher(unittest.TestCase):

    def test_compute_statistics(self):
        """test compute statistics
        """
        SETTINGS.training.vocab = ["hey", "fire", "fox"]
        SETTINGS.training.token_type = "word"
        SETTINGS.inference_engine.inference_sequence = [0, 1, 2]

        vocab = Vocab({"hey": 0, "fire": 1, "fox": 2}, oov_token_id=3, oov_word_repr='<OOV>')

        searcher = WordTranscriptSearcher(vocab)
        labeler = WordFrameLabeler(vocab)

        loader = WakeWordDatasetLoader()
        ds_kwargs = dict(sr=SETTINGS.audio.sample_rate, mono=SETTINGS.audio.use_mono, frame_labeler=labeler)
        test_train_ds, test_dev_ds, test_test_ds = loader.load_splits(Path("test/test_data"), **ds_kwargs)

        stitcher = WordStitcher(searcher, vocab=vocab)
        stitcher.stitch(test_train_ds)


if __name__ == '__main__':
    unittest.main()
