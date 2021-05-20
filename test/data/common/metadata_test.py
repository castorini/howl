import unittest

from howl.data.common.metadata import AudioClipMetadata


class TestAudioClipMetadata(unittest.TestCase):
    def test_instantiation(self):

        words = ["abc", "def"]
        timestamps = [100.0, 200.0]
        transcription = "hello world"
        metadata = AudioClipMetadata(
            phone_strings=None,
            words=words,
            phone_end_timestamps=timestamps,
            word_end_timestamps=timestamps,
            end_timestamps=timestamps,
            transcription=transcription,
        )

        self.assertEqual(str(metadata.path), ".")
        self.assertEqual(metadata.phone_strings, None)
        self.assertEqual(metadata.words[0], words[0])
        self.assertEqual(metadata.words[1], words[1])
        self.assertEqual(metadata.phone_end_timestamps[0], timestamps[0])
        self.assertEqual(metadata.phone_end_timestamps[1], timestamps[1])
        self.assertEqual(metadata.word_end_timestamps[0], timestamps[0])
        self.assertEqual(metadata.word_end_timestamps[1], timestamps[1])
        self.assertEqual(metadata.end_timestamps[0], timestamps[0])
        self.assertEqual(metadata.end_timestamps[1], timestamps[1])
        self.assertEqual(metadata.transcription, transcription)

        self.assertEqual(metadata.audio_id, "")
        self.assertEqual(metadata.phone_phrase, None)

        path = "abc.wav"
        phone_strings = ["hello", "world"]

        metadata = AudioClipMetadata(
            path=path,
            phone_strings=phone_strings,
            words=words,
            phone_end_timestamps=timestamps,
            word_end_timestamps=timestamps,
            end_timestamps=timestamps,
            transcription=transcription,
        )

        self.assertEqual(str(metadata.path), path)
        self.assertEqual(metadata.phone_strings[0], phone_strings[0])
        self.assertEqual(metadata.phone_strings[1], phone_strings[1])
        self.assertEqual(metadata.words[0], words[0])
        self.assertEqual(metadata.words[1], words[1])
        self.assertEqual(metadata.phone_end_timestamps[0], timestamps[0])
        self.assertEqual(metadata.phone_end_timestamps[1], timestamps[1])
        self.assertEqual(metadata.word_end_timestamps[0], timestamps[0])
        self.assertEqual(metadata.word_end_timestamps[1], timestamps[1])
        self.assertEqual(metadata.end_timestamps[0], timestamps[0])
        self.assertEqual(metadata.end_timestamps[1], timestamps[1])
        self.assertEqual(metadata.transcription, transcription)

        self.assertEqual(metadata.audio_id, path.split(".", 1)[0])
        phone_phrase = metadata.phone_phrase
        self.assertEqual(phone_phrase.phones[0].text, phone_strings[0])
        self.assertEqual(phone_phrase.phones[1].text, phone_strings[1])


if __name__ == "__main__":
    unittest.main()
