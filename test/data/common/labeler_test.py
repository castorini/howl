import unittest

from howl.data.common.labeler import WordFrameLabeler
from howl.data.common.metadata import AudioClipMetadata
from howl.data.common.vocab import Vocab
from howl.settings import SETTINGS


class TestWordFrameLabeler(unittest.TestCase):
    def test_word_frame_labeler(self):
        """test compute_frame_label"""
        SETTINGS.training.vocab = ["hello", "world"]
        SETTINGS.training.token_type = "word"
        SETTINGS.inference_engine.inference_sequence = [0, 1]

        vocab = Vocab({"Hello": 0, "World": 1}, oov_token_id=2, oov_word_repr="<OOV>")

        transcription = "hello world"
        timestamps = list(range(len(transcription)))  # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        metadata = AudioClipMetadata(
            phone_strings=None, phone_end_timestamps=timestamps, end_timestamps=timestamps, transcription=transcription,
        )

        labeler = WordFrameLabeler(vocab)
        frame_label_data = labeler.compute_frame_labels(metadata)

        self.assertEqual(len(frame_label_data.timestamp_label_map), 2)
        self.assertEqual(frame_label_data.timestamp_label_map[4.0], 0)
        self.assertEqual(frame_label_data.timestamp_label_map[10.0], 1)

        self.assertEqual(len(frame_label_data.start_timestamp), 2)
        self.assertEqual(frame_label_data.start_timestamp[0], (0, 0.0))
        self.assertEqual(frame_label_data.start_timestamp[1], (1, 5.0))

        self.assertEqual(len(frame_label_data.char_indices), 2)
        self.assertEqual(frame_label_data.char_indices[0], (0, list(range(5))))
        self.assertEqual(frame_label_data.char_indices[1], (1, list(range(6, 11))))


class TestPhoneticFrameLabeler(unittest.TestCase):
    def test_phonetic_frame_labeler(self):
        self.assertTrue(True)


if __name__ == "__main__":
    unittest.main()
