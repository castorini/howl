from pathlib import Path
from typing import Tuple

from howl.data.dataset.dataset import AudioDataset, DatasetSplit
from howl.utils.logger import Logger


class AudioDatasetLoader:
    """Load and create dataset instances from local files."""

    def __init__(self, name: str, dataset_path: Path):
        """Initialize AudioDatasetLoader for the given path.

        Args:
            name: unique name for the dataset loader
            dataset_path: location of the dataset
            logger = logger
        """
        self.name = name
        self.dataset_path = dataset_path
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Dataset path for {self.name} is invalid: {self.dataset_path}")

    def _load_dataset(self, dataset_split: DatasetSplit, **dataset_kwargs) -> AudioDataset:
        """Load dataset of given dataset_split.

        Args:
            dataset_split: dataset_split of the dataset to load
            **dataset_kwargs: other arguments passed to the dataset

        Returns:
            dataset for the given dataset_split
        """
        # pylint: disable=unused-argument
        Logger.info(f"Loading {dataset_split.value} split")
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

    def load_split(self, dataset_split: DatasetSplit, **dataset_kwargs) -> AudioDataset:
        """Load audio datasets of given split

        Args:
            dataset_split: dataset split to load
            dataset_kwargs: other arguments passed to individual dataset
        """
        return self._load_dataset(dataset_split, **dataset_kwargs)
