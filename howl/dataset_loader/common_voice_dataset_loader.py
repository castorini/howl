import logging
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from howl.data.common.metadata import AudioClipMetadata
from howl.data.dataset.dataset import AudioDataset, DatasetSplit
from howl.dataset.raw_audio_dataset import RawAudioDataset
from howl.dataset_loader.dataset_loader import AudioDatasetLoader


class CommonVoiceDatasetLoader(AudioDatasetLoader):
    """DatasetLoader for mozilla common-voice dataset"""

    # unique name which this dataset loader will be referred to
    NAME = "mozilla-cv"

    # name of the metafiles for each split
    METAFILE_MAPPING = {
        DatasetSplit.TRAINING: "train.tsv",
        DatasetSplit.DEV: "dev.tsv",
        DatasetSplit.TEST: "test.tsv",
    }

    def __init__(self, dataset_path: Path, logger: logging.Logger = None):
        """initialize CommonVoiceDatasetLoader for the given path

        Args:
            dataset_path: location of the dataset
            logger: logger
        """
        super().__init__(self.NAME, dataset_path=dataset_path, logger=logger)

    def _load_dataset(self, dataset_split: DatasetSplit, **dataset_kwargs) -> AudioDataset:
        """Load dataset of given dataset_split

        Args:
            dataset_split: dataset_split of the dataset to load
            **dataset_kwargs: other arguments passed to the dataset

        Returns:
            dataset for the given dataset_split
        """
        metafile_path = self.dataset_path / self.METAFILE_MAPPING[dataset_split]
        if not metafile_path.exists():
            raise FileNotFoundError(f"Metafile path for {dataset_split.value} split is invalid: {metafile_path}")
        data_frame = pd.read_csv(str(metafile_path), sep="\t", quoting=3, na_filter=False)
        metadata_list = []
        progress_bar = tqdm(data_frame.itertuples(), desc=f"loading {dataset_split.value} metadata")
        for tup in progress_bar:
            metadata_list.append(
                AudioClipMetadata(path=(self.dataset_path / "clips" / tup.path).absolute(), transcription=tup.sentence,)
            )
        return RawAudioDataset(metadata_list=metadata_list, dataset_split=dataset_split, **dataset_kwargs)
