import unittest

import torch

from howl.data.common.example import AudioClipExample, ClassificationClipExample, WakeWordClipExample
from howl.data.common.label import FrameLabelData
from howl.data.common.metadata import AudioClipMetadata


class TestAudioClipExample(unittest.TestCase):
    """Test case for AudioClipExample class"""

    def setUp(self):
        self.sample_rate = 100
        self.audio_data = torch.zeros(self.sample_rate)
        self.transcription = "hello world"
        self.metadata = AudioClipMetadata(transcription=self.transcription)

    def test_instantiation(self):
        """Test AudioClipExample instantiation"""
        sample = AudioClipExample(metadata=self.metadata, audio_data=self.audio_data, sample_rate=self.sample_rate)

        self.assertEqual(sample.metadata.transcription, self.transcription)
        self.assertTrue(torch.equal(sample.audio_data, self.audio_data))
        self.assertTrue(sample.sample_rate, self.sample_rate)

    def test_update_audio_data(self):
        """Test update_audio_data functionality"""
        new_audio_data = torch.ones(self.sample_rate)
        sample = AudioClipExample(metadata=self.metadata, audio_data=self.audio_data, sample_rate=self.sample_rate)

        updated_sample = sample.update_audio_data(new_audio_data)
        self.assertEqual(updated_sample.metadata.transcription, self.transcription)
        self.assertTrue(torch.equal(updated_sample.audio_data, new_audio_data))
        self.assertTrue(updated_sample.sample_rate, self.sample_rate)

        updated_sample = sample.update_audio_data(new_audio_data, new=True)
        self.assertEqual(updated_sample.metadata.transcription, "")
        self.assertTrue(torch.equal(updated_sample.audio_data, new_audio_data))
        self.assertTrue(sample.sample_rate, self.sample_rate)


class TestClassificationClipExample(unittest.TestCase):
    """Test case for ClassificationClipExample class"""

    def setUp(self):
        self.sample_rate = 100
        self.audio_data = torch.zeros(self.sample_rate)
        self.transcription = "hello world"
        self.metadata = AudioClipMetadata(transcription=self.transcription)
        self.label = 0

    def test_instantiation(self):
        """Test ClassificationClipExample instantiation"""
        sample = ClassificationClipExample(
            metadata=self.metadata, audio_data=self.audio_data, sample_rate=self.sample_rate, label=self.label
        )

        self.assertEqual(sample.metadata.transcription, self.transcription)
        self.assertTrue(torch.equal(sample.audio_data, self.audio_data))
        self.assertTrue(sample.sample_rate, self.sample_rate)
        self.assertEqual(sample.label, self.label)

    def test_update_audio_data(self):
        """Test update_audio_data functionality"""
        new_audio_data = torch.ones(self.sample_rate)
        sample = ClassificationClipExample(
            metadata=self.metadata, audio_data=self.audio_data, sample_rate=self.sample_rate, label=self.label
        )

        updated_sample = sample.update_audio_data(new_audio_data)
        self.assertEqual(updated_sample.metadata.transcription, self.transcription)
        self.assertTrue(torch.equal(updated_sample.audio_data, new_audio_data))
        self.assertTrue(updated_sample.sample_rate, self.sample_rate)


class TestWakeWordClipExample(unittest.TestCase):
    """Test case for WakeWordClipExample class"""

    def setUp(self):
        self.sample_rate = 100
        self.audio_data = torch.zeros(self.sample_rate)
        self.transcription = "hello world"
        self.metadata = AudioClipMetadata(transcription=self.transcription)
        self.timestamp = 500.0
        self.label = 0
        self.label_data = FrameLabelData(
            timestamp_label_map={self.timestamp: self.label}, start_timestamp=[], char_indices=[]
        )

    def test_instantiation(self):
        """Test WakeWordClipExample instantiation"""
        sample = WakeWordClipExample(
            metadata=self.metadata, audio_data=self.audio_data, sample_rate=self.sample_rate, label_data=self.label_data
        )

        self.assertEqual(sample.metadata.transcription, self.transcription)
        self.assertTrue(torch.equal(sample.audio_data, self.audio_data))
        self.assertTrue(sample.sample_rate, self.sample_rate)
        self.assertEqual(sample.label_data.timestamp_label_map[self.timestamp], self.label)
        self.assertEqual(len(sample.label_data.start_timestamp), 0)
        self.assertEqual(len(sample.label_data.char_indices), 0)

    def test_update_audio_data(self):
        """Test update_audio_data functionality"""
        new_audio_data = torch.ones(self.sample_rate)
        sample = WakeWordClipExample(
            metadata=self.metadata, audio_data=self.audio_data, sample_rate=self.sample_rate, label_data=self.label_data
        )

        scale = 2
        bias = 100.0

        updated_sample = sample.update_audio_data(new_audio_data, scale=scale, bias=bias)
        self.assertEqual(updated_sample.metadata.transcription, self.transcription)
        self.assertTrue(torch.equal(updated_sample.audio_data, new_audio_data))
        self.assertTrue(updated_sample.sample_rate, self.sample_rate)
        self.assertEqual(updated_sample.label_data.timestamp_label_map[self.timestamp * scale + bias], self.label)
        self.assertEqual(len(updated_sample.label_data.start_timestamp), 0)
        self.assertEqual(len(updated_sample.label_data.char_indices), 0)


if __name__ == "__main__":
    unittest.main()
