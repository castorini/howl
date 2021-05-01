from collections import defaultdict
from pathlib import Path
from typing import List, Tuple

from howl.data.common.metadata import AudioClipMetadata
from howl.data.dataset.dataset import AudioClassificationDataset, DatasetType
from howl.data.dataset_loader.dataset_loader import RegisteredPathDatasetLoader

__all__ = [
    "GoogleSpeechCommandsDatasetLoader",
]


class GoogleSpeechCommandsDatasetLoader(RegisteredPathDatasetLoader, name="gsc"):
    def __init__(self, vocab: List[str] = None, use_bg_noise: bool = False):
        self.vocab = vocab
        self.use_bg_noise = use_bg_noise

    def load_splits(
        self, path: Path, **dataset_kwargs
    ) -> Tuple[
        AudioClassificationDataset,
        AudioClassificationDataset,
        AudioClassificationDataset,
    ]:
        def load(set_type):
            metadata_list = []
            for path in all_list:
                key = str(Path(path.parent.name) / path.name)
                if file_map[key] != set_type:
                    continue
                metadata_list.append(
                    AudioClipMetadata(
                        path=path.absolute(), transcription=path.parent.name
                    )
                )
            return AudioClassificationDataset(
                metadata_list=metadata_list,
                label_map=label_map,
                set_type=set_type,
                **dataset_kwargs
            )

        file_map = defaultdict(lambda: DatasetType.TRAINING)
        with (path / "testing_list.txt").open() as f:
            file_map.update({k: DatasetType.TEST for k in f.read().split("\n")})
        with (path / "validation_list.txt").open() as f:
            file_map.update({k: DatasetType.DEV for k in f.read().split("\n")})
        all_list = list(path.glob("*/*.wav"))
        if not self.use_bg_noise:
            all_list = list(filter(lambda x: "noise" not in str(x), all_list))
        folders = sorted(list(path.glob("*/")))
        vocab = [x.name for x in folders] if self.vocab is None else self.vocab
        label_map = defaultdict(lambda: len(vocab))
        label_map.update({k: idx for idx, k in enumerate(vocab)})
        return load(DatasetType.TRAINING), load(DatasetType.DEV), load(DatasetType.TEST)
