import unittest

from howl.data.dataset.dataset import DatasetSplit
from howl.dataset.audio_dataset_constants import AudioDatasetType, SampleType
from howl.dataset_loader.howl_audio_dataset_loader import HowlAudioDatasetLoader
from howl.utils import test_utils


class HowlAudioDatasetTest(unittest.TestCase):
    """Test case for HowlAudioDataset"""

    def _load_sample(self, dataset_type: AudioDatasetType):
        """Load a sample of given type and split"""

        # region load datasets
        dataset_path = test_utils.howl_audio_datasets_path() / test_utils.WAKEWORD / SampleType.POSITIVE.value
        dataset_loader = HowlAudioDatasetLoader(dataset_type, dataset_path)
        if dataset_type == AudioDatasetType.RAW:
            ds_kwargs = dict(dataset_split=DatasetSplit.TRAINING,)
        elif dataset_type == AudioDatasetType.ALIGNED:
            ds_kwargs = dict(dataset_split=DatasetSplit.TRAINING, labeler=test_utils.frame_labeller(test_utils.VOCAB),)
        dataset = dataset_loader.load_split(**ds_kwargs)

        return dataset[0]

    def test_raw_audio_dataset(self):
        """Validate the logic for loading a raw sample from howl audio dataset"""
        sample = self._load_sample(AudioDatasetType.RAW)
        self.assertGreater(len(sample.audio_data), 0)
        self.assertIsNone(sample.label)
        self.assertFalse(sample.labelled)
        self.assertIsNotNone(sample.metadata.audio_id)
        self.assertIsNotNone(sample.metadata.path)
        self.assertIsNotNone(sample.metadata.transcription)
        self.assertIsNone(sample.metadata.end_timestamps)
        self.assertIsNone(sample.metadata.phone_end_timestamps)
        self.assertIsNone(sample.metadata.phone_phrase)
        self.assertIsNone(sample.metadata.phone_strings)
        self.assertIsNone(sample.metadata.words)

    def test_aligned_audio_dataset(self):
        """Validate the logic for loading a raw sample from howl audio dataset"""
        sample = self._load_sample(AudioDatasetType.ALIGNED)
        self.assertGreater(len(sample.audio_data), 0)
        self.assertIsNotNone(sample.label)
        self.assertEqual(len(sample.label.char_indices), 0)
        self.assertEqual(len(sample.label.start_timestamp), 0)
        self.assertEqual(len(sample.label.timestamp_label_map), len(test_utils.WAKEWORD))
        self.assertTrue(sample.labelled)
        self.assertIsNotNone(sample.metadata.audio_id)
        self.assertIsNotNone(sample.metadata.path)
        self.assertIsNotNone(sample.metadata.transcription)
        self.assertIsNotNone(sample.metadata.end_timestamps)
        self.assertIsNone(sample.metadata.phone_end_timestamps)
        self.assertIsNone(sample.metadata.phone_phrase)
        self.assertIsNone(sample.metadata.phone_strings)
        self.assertIsNone(sample.metadata.words)
