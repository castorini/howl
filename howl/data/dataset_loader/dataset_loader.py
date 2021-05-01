import json
import logging
from pathlib import Path
from typing import Tuple

from howl.data.common.metadata import AudioClipMetadata
from howl.data.dataset.dataset import (
    AudioClipDataset,
    AudioDataset,
    DatasetType,
    WakeWordDataset,
)
from howl.utils.class_registry import ClassRegistry

__all__ = [
    "PathDatasetLoader",
    "RegisteredPathDatasetLoader",
    "MetadataLoaderMixin",
    "AudioClipDatasetLoader",
    "WakeWordDatasetLoader",
    "RecursiveNoiseDatasetLoader",
]


class PathDatasetLoader:
    def load_splits(
        self, path: Path, **dataset_kwargs
    ) -> Tuple[AudioDataset, AudioDataset, AudioDataset]:
        raise NotImplementedError


class RegisteredPathDatasetLoader(PathDatasetLoader, ClassRegistry):
    registered_map = {}


class MetadataLoaderMixin:
    dataset_class = None
    metadata_class = None
    default_prefix = ""

    def load_splits(self, path: Path, prefix: str = None, **dataset_kwargs):
        def load(jsonl_name):
            metadata_list = []
            with open(jsonl_name) as f:
                for json_str in iter(f.readline, ""):
                    metadata = self.metadata_class(**json.loads(json_str))
                    metadata.path = (path / "audio" / metadata.path).absolute()
                    metadata_list.append(metadata)
                return metadata_list

        if prefix is None:
            prefix = self.default_prefix
        logging.info(f"Loading flat dataset from {path}...")
        training_path = (
            path / f"{prefix}metadata-{DatasetType.TRAINING.name.lower()}.jsonl"
        )
        dev_path = path / f"{prefix}metadata-{DatasetType.DEV.name.lower()}.jsonl"
        test_path = path / f"{prefix}metadata-{DatasetType.TEST.name.lower()}.jsonl"
        return (
            self.dataset_class(
                metadata_list=load(training_path),
                set_type=DatasetType.TRAINING,
                **dataset_kwargs,
            ),
            self.dataset_class(
                metadata_list=load(dev_path), set_type=DatasetType.DEV, **dataset_kwargs
            ),
            self.dataset_class(
                metadata_list=load(test_path),
                set_type=DatasetType.TEST,
                **dataset_kwargs,
            ),
        )


class AudioClipDatasetLoader(
    MetadataLoaderMixin, RegisteredPathDatasetLoader, name="clip"
):
    dataset_class = AudioClipDataset
    metadata_class = AudioClipMetadata


class WakeWordDatasetLoader(MetadataLoaderMixin, PathDatasetLoader):
    default_prefix = "aligned-"
    dataset_class = WakeWordDataset
    metadata_class = AudioClipMetadata


class RecursiveNoiseDatasetLoader:
    def load(self, path: Path, **dataset_kwargs) -> AudioClipDataset:
        wav_names = path.glob("**/*.wav")
        metadata_list = [
            AudioClipMetadata(path=filename.absolute(), transcription="")
            for filename in wav_names
        ]
        return AudioClipDataset(
            metadata_list=metadata_list, set_type=DatasetType.TRAINING, **dataset_kwargs
        )
