from pathlib import Path

import pandas as pd
from tqdm import tqdm

from howl.data.common.metadata import AudioClipMetadata
from howl.data.dataset.dataset import AudioDataset, DatasetSplit
from howl.dataset.audio_dataset_constants import AudioDatasetType
from howl.dataset.howl_audio_dataset import HowlAudioDataset
from howl.dataset_loader.dataset_loader import AudioDatasetLoader


class CommonVoiceDatasetLoader(AudioDatasetLoader):
    """DatasetLoader for mozilla common-voice dataset"""

    # name of the metafiles for each split
    METAFILE_MAPPING = {
        DatasetSplit.TRAINING: "train.tsv",
        DatasetSplit.DEV: "dev.tsv",
        DatasetSplit.TEST: "test.tsv",
    }

    def __init__(self, dataset_path: Path):
        """initialize CommonVoiceDatasetLoader for the given path

        Args:
            dataset_path: location of the dataset
        """
        super().__init__(AudioDatasetType.COMMON_VOICE.value, dataset_path=dataset_path)

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
        return HowlAudioDataset(metadata_list=metadata_list, dataset_split=dataset_split, **dataset_kwargs)
