
import random
import unittest
from pathlib import Path

import torch
from howl.data.dataset import (AudioClipExample, AudioClipMetadata,
                               AudioDatasetWriter, DatasetType,
                               WakeWordDatasetLoader, WordFrameLabeler)
from howl.data.searcher import WordTranscriptSearcher
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

        vocab = Vocab({"hey": 0, "fire": 1, "fox": 2}, oov_token_id=3, oov_word_repr='<OOV>')

        searcher = WordTranscriptSearcher(vocab)
        labeler = WordFrameLabeler(vocab)

        loader = WakeWordDatasetLoader()
        ds_kwargs = dict(sr=SETTINGS.audio.sample_rate, mono=SETTINGS.audio.use_mono, frame_labeler=labeler)

        test_dataset_path = Path("test/test_data")
        test_train_ds, test_dev_ds, test_test_ds = loader.load_splits(test_dataset_path, **ds_kwargs)

        stitcher = WordStitcher(searcher, vocab=vocab)
        stitcher.stitch(test_train_ds)

        stitched_train_ds, stitched_dev_ds, stitched_test_ds = stitcher.load_splits(0.5, 0.2, 0.2)

        stitched_dataset_path = test_dataset_path / "stitched_dataset"
        stitched_dataset_path.mkdir(exist_ok=True)

        for ds in stitched_train_ds, stitched_dev_ds, stitched_test_ds:
            try:
                AudioDatasetWriter(ds, prefix='aligned-').write(stitched_dataset_path)
            except KeyboardInterrupt:
                print('Skipping...')
                pass

        print(stitched_train_ds[0].metadata.path)
        print(stitched_dev_ds[0].metadata.path)
        print(stitched_test_ds[0].metadata.path)

        print('loaded')

        reloated_stitched_train_ds, reloated_stitched_dev_ds, reloated_stitched_test_ds = loader.load_splits(
            stitched_dataset_path, **ds_kwargs)

        print(reloated_stitched_train_ds[0].metadata.path)
        print(reloated_stitched_dev_ds[0].metadata.path)
        print(reloated_stitched_test_ds[0].metadata.path)


if __name__ == '__main__':
    unittest.main()
