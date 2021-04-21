import tempfile
import unittest
from pathlib import Path

import pytest
import torch

from howl.data.dataset import (
    AudioClipDataset,
    AudioClipExample,
    AudioClipMetadata,
    AudioDatasetWriter,
    DatasetType,
)
from howl.settings import SETTINGS


class TestDataset(AudioClipDataset):
    """Sample dataset for testing"""

    __test__ = False

    def __init__(self, num_samples: int):

        audio_data = torch.zeros(SETTINGS.audio.sample_rate)

        self.samples = []
        metadata_list = []
        for i in range(num_samples):
            audio_id = f"test_sample_{i}"
            metadata = AudioClipMetadata(path=audio_id, transcription=audio_id)
            sample = AudioClipExample(metadata=metadata, audio_data=audio_data, sample_rate=SETTINGS.audio.sample_rate)
            metadata_list.append(metadata)
            self.samples.append(sample)

        super().__init__(metadata_list, sr=SETTINGS.audio.sample_rate, set_type=DatasetType.TRAINING)


class TestAudioDatasetWriter(unittest.TestCase):
    @pytest.mark.skip(reason="write function fails due to missing audio data")
    def test_write(self):
        """test multiprocessing functionality for writing a huge dataset
        """

        dataset = TestDataset(500)

        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_dir = Path(temp_dir)
            AudioDatasetWriter(dataset).write(dataset_dir)

            self.assertTrue(dataset_dir.exists())
            self.assertTrue((dataset_dir / "audio").exists())
            self.assertTrue((dataset_dir / "metadata-training.jsonl").exists())

            # TODO:: make sure the contents of these folders are valid


if __name__ == "__main__":
    unittest.main()
