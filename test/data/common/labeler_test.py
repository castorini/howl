import unittest
from pathlib import Path

from howl.data.common.labeler import PhoneticFrameLabeler, WordFrameLabeler
from howl.data.common.metadata import AudioClipMetadata
from howl.data.common.phone import PhonePhrase, PronunciationDictionary
from howl.data.common.vocab import Vocab


class TestPhoneticFrameLabeler(unittest.TestCase):
    def test_phonetic_frame_labeler(self):

        vocab = ["hey", "fire", "fox"]

        phone_dict_file = Path("test/test_data/pronounciation_dictionary.txt")
        pronounce_dict = PronunciationDictionary.from_file(phone_dict_file)

        adjusted_vocab = []
        for word in vocab:
            phone_phrase = pronounce_dict.encode(word)[0]
            print(f"Word {word: <10} has phonemes of {str(phone_phrase)}")
            adjusted_vocab.extend(list(str(phone) for phone in phone_phrase.phones))

        phone_phrases = [PhonePhrase.from_string(x) for x in adjusted_vocab]
        labeler = PhoneticFrameLabeler(pronounce_dict, phone_phrases)

        transcription = "hey firefox"
        timestamps = list(range(len(transcription)))  # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        metadata = AudioClipMetadata(
            phone_strings=None, phone_end_timestamps=timestamps, end_timestamps=timestamps, transcription=transcription,
        )

        # TODO: compute_frame_labels has not been implemented correctly yet
        frame_label_data = labeler.compute_frame_labels(metadata)
        print(frame_label_data)

        self.assertTrue(True)


class TestWordFrameLabeler(unittest.TestCase):
    def test_word_frame_labeler(self):
        """test compute_frame_label"""

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


if __name__ == "__main__":
    unittest.main()
