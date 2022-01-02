import unittest

import torch

from howl.data.common.label import FrameLabelData
from howl.data.common.metadata import AudioClipMetadata
from howl.data.common.sample import Sample


class SampleTest(unittest.TestCase):
    """Test case for Sample class"""

    def setUp(self):
        self.sample_rate = 100
        self.audio_data = torch.zeros(self.sample_rate)
        self.transcription = "hello world"
        self.metadata = AudioClipMetadata(transcription=self.transcription)
        self.label = FrameLabelData(
            timestamp_label_map=[1.0, 0], start_timestamp=[(0, 1.0)], char_indices=[(1, [0, 1, 2])]
        )

    def test_sample(self):
        """Test instantiation of a sample"""
        unlabelled_sample = Sample(metadata=self.metadata, audio_data=self.audio_data, sample_rate=self.sample_rate)
        self.assertEqual(unlabelled_sample.metadata.transcription, self.transcription)
        self.assertTrue(torch.equal(unlabelled_sample.audio_data, self.audio_data))
        self.assertTrue(unlabelled_sample.sample_rate, self.sample_rate)
        self.assertFalse(unlabelled_sample.labelled)

        labelled_sample = Sample(
            metadata=self.metadata, audio_data=self.audio_data, sample_rate=self.sample_rate, label=self.label
        )
        self.assertEqual(labelled_sample.metadata.transcription, self.transcription)
        self.assertTrue(torch.equal(labelled_sample.audio_data, self.audio_data))
        self.assertTrue(labelled_sample.sample_rate, self.sample_rate)
        self.assertTrue(labelled_sample.labelled)
        self.assertTrue(labelled_sample.label, self.label)
