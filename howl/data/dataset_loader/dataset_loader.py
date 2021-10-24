import logging
from pathlib import Path
from typing import Tuple

from howl.data.dataset.dataset import AudioDataset, DatasetSplit
from howl.utils import logging_utils


class AudioDatasetLoader:
    def __init__(self, name: str, dataset_path: Path, logger: logging.Logger = None):
        """Load audio datasets from the provided path

        Args:
            name: unique name for the dataset loader
            dataset_path: location of the dataset
            logger = logger
        """
        self.name = name
        self.dataset_path = dataset_path
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Dataset path for {self.name} is invalid: {self.dataset_path}")

        self.logger = logger
        if self.logger is None:
            self.logger = logging_utils.setup_logger(f"AudioDatasetLoader({self.name})")

    def _load_dataset(self, split: DatasetSplit, **dataset_kwargs) -> AudioDataset:
        """Load dataset of given split

        Args:
            split: split of the dataset to load
            **dataset_kwargs: other arguments passed to the dataset

        Returns:
            dataset for the given split
        """
        self.logger.info(f"Loading {split.value} split")
        raise NotImplementedError("Not yet implemented for {}".format(self.__class__.__name__))

    def load_splits(self, **dataset_kwargs) -> Tuple[AudioDataset, AudioDataset, AudioDataset]:
        """Load train/dev/test audio datasets

        Args:
            dataset_kwargs: other arguments passed to individual dataset
        """
        return (
            self._load_dataset(DatasetSplit.TRAINING, **dataset_kwargs),
            self._load_dataset(DatasetSplit.DEV, **dataset_kwargs),
            self._load_dataset(DatasetSplit.TEST, **dataset_kwargs),
        )
