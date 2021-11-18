import json
import logging
from pathlib import Path

from tqdm import tqdm

from howl.data.common.metadata import AudioClipMetadata
from howl.data.dataset.dataset import AudioDataset, DatasetSplit
from howl.dataset.raw_audio_dataset import RawAudioDataset
from howl.dataset_loader.dataset_loader import AudioDatasetLoader


class RawAudioDatasetLoader(AudioDatasetLoader):
    """DatasetLoader for raw audio dataset"""

    # unique name which this dataset loader will be referred to
    NAME = "raw"

    def __init__(self, dataset_path: Path, logger: logging.Logger = None):
        """Initialize AudioDatasetLoader for the given path.

        Args:
            dataset_path: location of the dataset
            logger = logger
        """
        super().__init__(self.NAME, dataset_path=dataset_path, logger=logger)

    def _load_dataset(self, dataset_split: DatasetSplit, **dataset_kwargs) -> AudioDataset:
        """Load raw audio dataset of given dataset_split

        Args:
            dataset_split: dataset_split of the dataset to load
            **dataset_kwargs: other arguments passed to the dataset

        Returns:
            raw audio dataset for the given dataset_split
        """
        metadata_file_path = self.dataset_path / RawAudioDataset.METADATA_FILE_NAME_TEMPLATE.format(
            dataset_split=dataset_split.value
        )
        if not metadata_file_path.exists():
            raise FileNotFoundError(f"Metafile path for {dataset_split.value} is missing: {metadata_file_path}")

        metadata_list = []
        with open(metadata_file_path) as metadata_file:
            for json_str in tqdm(iter(metadata_file.readline, ""), desc=f"loading {metadata_file_path}"):
                metadata = AudioClipMetadata(**json.loads(json_str))
                metadata.path = self.dataset_path / "audio" / metadata.path
                metadata_list.append(metadata)

        return RawAudioDataset(metadata_list=metadata_list, dataset_split=dataset_split, **dataset_kwargs)
