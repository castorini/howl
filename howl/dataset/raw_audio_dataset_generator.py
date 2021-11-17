import logging
from copy import deepcopy
from enum import Enum, unique
from pathlib import Path

from howl.context import InferenceContext
from howl.data.common.metadata import AudioClipMetadata
from howl.data.dataset.dataset_writer import AudioDatasetWriter
from howl.dataset_loader.dataset_loader_factory import DatasetLoaderType, get_dataset_loader
from howl.settings import SETTINGS
from howl.utils import hash_utils, logging_utils


@unique
class SampleType(str, Enum):
    """String based Enum for positive/negative sample type"""

    POSITIVE = "positive"
    NEGATIVE = "negative"


class RawAudioDatasetGenerator:
    """Generate audio dataset"""

    def __init__(self, input_dataset_path: Path, dataset_loader_type: DatasetLoaderType, logger: logging.Logger = None):
        """initialize RawAudioDatasetGenerator by loading dataset from the given path

        Args:
            input_dataset_path: location of the dataset
            dataset_loader_type: type of dataset loader to use
            logger: logger
        """
        self.input_dataset_path = input_dataset_path
        if not self.input_dataset_path.exists():
            raise FileNotFoundError(f"Dataset path is invalid: {self.input_dataset_path}")

        self.logger = logger
        if self.logger is None:
            self.logger = logging_utils.setup_logger(self.__class__.__name__)

        self.dataset_loader_type = dataset_loader_type
        self.dataset_loader = get_dataset_loader(dataset_loader_type, Path(input_dataset_path), self.logger)
        self.inference_ctx = InferenceContext(SETTINGS.training.vocab, token_type=SETTINGS.training.token_type)

        ds_kwargs = dict(sample_rate=SETTINGS.audio.sample_rate, mono=SETTINGS.audio.use_mono)
        self.train_ds, self.dev_ds, self.test_ds = self.dataset_loader.load_splits(**ds_kwargs)

    def filter_fn(self, metadata: AudioClipMetadata, sample_type: SampleType, percentage: int = 100):
        """Distribute given audio data into different bucket based on the configuration and
        return True if it needs to be included

        Args:
            metadata: metadata of the audio sample
            sample_type: target sample type to filter
            percentage: percentage of dataset to consider

        Returns:
            True if it needs to be included for the dataset
        """
        bucket = hash_utils.sha256_int(metadata.path.stem) % 100
        include_sample = False
        if bucket < percentage:
            if sample_type == SampleType.POSITIVE:
                # select all sample whose transcript contains at least one of the vocabs
                include_sample = self.inference_ctx.searcher.contains_any(metadata.transcription.lower())
            elif sample_type == SampleType.NEGATIVE:
                # drop the samples whose transcript is wakeword
                include_sample = not self.inference_ctx.searcher.search(metadata.transcription.lower())
            else:
                raise RuntimeError(f"Invalid sample type provided: {sample_type}")

        return include_sample

    def generate_datasets(
        self, dataset_path: Path, sample_type: SampleType, percentage: int = 100, print_statistics: bool = True
    ):
        """Filter target samples from the loaded datasets and save datasets to the given path

        Args:
            dataset_path: path of the generated dataset
            sample_type: target sample type to filter
            percentage: percentage of dataset to consider
            print_statistics: if True, compute statistic of the dataset for the vocab
        """

        self.logger.info(f"Generating {sample_type.value} dataset using {percentage}% of the data")

        if self.dataset_loader_type == DatasetLoaderType.COMMON_VOICE_DATASET_LOADER:
            predicate_fn_kwargs = dict(sample_type=sample_type, percentage=percentage)
            train_ds = deepcopy(self.train_ds).filter(self.filter_fn, **predicate_fn_kwargs)
            dev_ds = deepcopy(self.dev_ds).filter(self.filter_fn, **predicate_fn_kwargs)
            test_ds = deepcopy(self.test_ds).filter(self.filter_fn, **predicate_fn_kwargs)

        word_searcher = None
        if self.inference_ctx.token_type == "word":
            word_searcher = self.inference_ctx.searcher

        for dataset in train_ds, dev_ds, test_ds:
            if print_statistics:
                dataset.print_stats(self.logger, word_searcher=word_searcher, compute_length=True)
            self.logger.info(f"Generating {dataset.dataset_split.value} dataset")
            AudioDatasetWriter(dataset, logger=self.logger).write(dataset_path)
