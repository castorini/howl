import copy
from pathlib import Path

from howl.data.common.labeler import FrameLabeler
from howl.data.common.vocab import Vocab
from howl.data.dataset.dataset import DatasetSplit
from howl.data.dataset.dataset_writer import AudioDatasetMetadataWriter
from howl.data.stitcher import WordStitcher
from howl.dataset.audio_dataset_constants import AudioDatasetType
from howl.dataset.howl_audio_dataset import HowlAudioDataset
from howl.dataset_loader.howl_audio_dataset_loader import HowlAudioDatasetLoader
from howl.settings import SETTINGS
from howl.utils.logger import Logger


class StitchedAudioDatasetGenerator:
    """Using aligned howl audio dataset, generate stitched howl audio dataset"""

    def __init__(
        self,
        aligned_audio_dataset_path: Path,
        vocab: Vocab,
        max_num_training_samples: int,
        max_num_dev_samples: int,
        max_num_test_samples: int,
        validate_stitched_sample: bool,
        sample_rate: int = SETTINGS.audio.sample_rate,
        mono: bool = SETTINGS.audio.use_mono,
        labeller: FrameLabeler = None,
    ):
        """initialize StitchedAudioDatasetGenerator by instantiating a stitcher

        Args:
            aligned_audio_dataset_path: location of the dataset
            vocab (Vocab): vocab containing wakeword
            max_num_training_samples: maximum number of stitched training audio sample to generate
            max_num_dev_samples: maximum number of stitched dev audio sample to generate
            max_num_test_samples: maximum number of stitched test audio sample to generate
            validate_stitched_sample: run additional validation on the stitched samples
            sample_rate: sample rate of the audio file
            mono: if True, the audio file will be loaded assuming the data is mono channel
            labeller (FrameLabeler): labeler to use for loading aligned dataset
        """

        # load aligned datasets
        self.vocab = vocab
        self.dataset_loader = HowlAudioDatasetLoader(AudioDatasetType.ALIGNED, aligned_audio_dataset_path)
        self.validate_stitched_sample = validate_stitched_sample
        self.base_ds_kwargs = dict(sample_rate=sample_rate, mono=mono, labeler=labeller)
        self.aligned_audio_dataset_path = aligned_audio_dataset_path

        self.max_num_samples = {
            DatasetSplit.TRAINING: max_num_training_samples,
            DatasetSplit.DEV: max_num_dev_samples,
            DatasetSplit.TEST: max_num_test_samples,
        }

    def generate_datasets(self):
        """Generate stitched dataset from aligned dataset"""
        self._generate_dataset(DatasetSplit.TRAINING)
        self._generate_dataset(DatasetSplit.DEV)
        self._generate_dataset(DatasetSplit.TEST)

    def _generate_dataset(self, dataset_split: DatasetSplit):
        ds_kwargs = copy.deepcopy(self.base_ds_kwargs)
        ds_kwargs["dataset_split"] = dataset_split
        aligned_dataset = self.dataset_loader.load_split(**ds_kwargs)

        if len(self.vocab) <= 1:
            Logger.warning(f"Word stitching require at least two words: {self.vocab}")
            return

        stitcher = WordStitcher(vocab=self.vocab, validate_stitched_sample=self.validate_stitched_sample)

        audio_dir = self.aligned_audio_dataset_path / HowlAudioDataset.DIR_AUDIO
        audio_dir.mkdir(exist_ok=True)
        audio_sample_filename_template = dataset_split.value
        audio_sample_filename_template += "_{sample_idx}"
        stitcher.generate_stitched_audio_samples(
            self.max_num_samples[dataset_split],
            audio_dir,
            aligned_dataset,
            audio_sample_filename_template=audio_sample_filename_template,
        )

        with AudioDatasetMetadataWriter(
            self.aligned_audio_dataset_path, AudioDatasetType.STITCHED, dataset_split
        ) as metadata_writer:
            for sample in stitcher.stitched_samples:
                metadata_writer.write(sample.metadata)
