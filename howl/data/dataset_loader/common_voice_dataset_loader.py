import logging
from pathlib import Path

import pandas as pd

from howl.data.common.metadata import AudioClipMetadata
from howl.data.dataset.dataset import AudioClipDataset, AudioDataset, DatasetSplit
from howl.data.dataset_loader.dataset_loader import AudioDatasetLoader


class CommonVoiceDatasetLoader(AudioDatasetLoader):
    # unique name which this dataset loader will be referred to
    NAME = "mozilla-cv"

    # name of the metafiles for each split
    METAFILE_MAPPING = {
        DatasetSplit.TRAINING: "train.tsv",
        DatasetSplit.DEV: "dev.tsv",
        DatasetSplit.TEST: "test.tsv",
    }

    def __init__(self, dataset_path: Path, logger: logging.Logger = None):
        """DatasetLoader for mozilla common-voice dataset

        Args:
            dataset_path: location of the dataset
            logger: logger
        """
        super().__init__(self.NAME, dataset_path=dataset_path, logger=logger)

    def _load_dataset(self, split: DatasetSplit, **dataset_kwargs) -> AudioDataset:
        """Load dataset of given split

        Args:
            split: split of the dataset to load
            **dataset_kwargs: other arguments passed to the dataset

        Returns:
            dataset for the given split
        """
        metafile_path = self.dataset_path / self.METAFILE_MAPPING[split]
        if not metafile_path.exists():
            raise FileNotFoundError(f"Metafile path for {split.value} split is invalid: {metafile_path}")
        df = pd.read_csv(str(metafile_path), sep="\t", quoting=3, na_filter=False)
        metadata_list = []
        for tup in df.itertuples():
            metadata_list.append(
                AudioClipMetadata(path=(self.dataset_path / "clips" / tup.path).absolute(), transcription=tup.sentence,)
            )
        return AudioClipDataset(metadata_list=metadata_list, split=split, **dataset_kwargs)
