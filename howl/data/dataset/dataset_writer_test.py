import json
import os
import tempfile
import unittest
from pathlib import Path

from howl.data.dataset.dataset import DatasetType
from howl.data.dataset.dataset_writer import AudioDatasetWriter
from howl.utils import test_utils


class DatasetWriterTest(unittest.TestCase):
    """Test case for DatasetWriter"""

    def test_writing_dataset(self):
        """test write functionality of the dataset writer"""

        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_path = Path(temp_dir) / "dataset"

            audio_dataset = test_utils.TestDataset(metadata_list=[], dataset_split=DatasetType.TRAINING)
            AudioDatasetWriter(audio_dataset).write(dataset_path)

            self.assertTrue(dataset_path.exists())

            audio_dir_path = dataset_path / "audio"
            self.assertTrue(audio_dir_path.exists())

            metadata_json_file_path = dataset_path / "metadata-training.jsonl"
            self.assertTrue(metadata_json_file_path.exists())

            expected_keys = [
                "path",
                "phone_strings",
                "words",
                "phone_end_timestamps",
                "end_timestamps",
                "transcription",
            ]

            with open(metadata_json_file_path, "r") as metadata_json_file:
                for metadata in audio_dataset.metadata_list:
                    self.assertTrue(os.path.exists(metadata.path))

                    metadata_json = json.loads(metadata_json_file.readline())
                    for key in expected_keys:
                        self.assertIn(key, metadata_json)
                    self.assertEqual(metadata.path.name, metadata_json["path"])

                    transcription_file_path = metadata.path.with_suffix(".lab")
                    self.assertTrue(os.path.exists(transcription_file_path))
                    with open(transcription_file_path, "r") as transcription_file:
                        self.assertEqual(transcription_file.readline().strip(), metadata.transcription)
                        self.assertEqual(metadata_json["transcription"], metadata.transcription)
