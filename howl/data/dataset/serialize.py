import logging
from copy import deepcopy
from pathlib import Path
from typing import TypeVar

import soundfile
from tqdm import tqdm

from howl.data.common.metadata import AudioClipMetadata
from howl.data.dataset.dataset import AudioClipDataset, DatasetType
from howl.utils.audio import silent_load

__all__ = [
    "AudioDatasetWriter",
    "AudioDatasetMetadataWriter",
]


class AudioDatasetMetadataWriter:
    def __init__(
        self,
        dataset_path: Path,
        set_type: DatasetType,
        prefix: str = "",
        mode: str = "w",
    ):
        self.filename = str(
            dataset_path / f"{prefix}metadata-{set_type.name.lower()}.jsonl"
        )
        self.mode = mode

    def __enter__(self):
        self.f = open(self.filename, self.mode)
        return self

    def write(self, metadata: AudioClipMetadata):
        metadata = deepcopy(metadata)
        with metadata.path.with_suffix(".lab").open("w") as f:
            f.write(f"{metadata.transcription}\n")
        metadata.path = metadata.path.name
        self.f.write(metadata.json() + "\n")

    def __exit__(self, *args):
        self.f.close()


T = TypeVar("T", bound=AudioClipDataset)


class AudioDatasetWriter:
    def __init__(
        self,
        dataset: AudioClipDataset,
        prefix: str = "",
        mode: str = "w",
        print_progress: bool = True,
    ):
        self.dataset = dataset
        self.print_progress = print_progress
        self.mode = mode
        self.prefix = prefix

    def write(self, folder: Path):
        def process(metadata: AudioClipMetadata):
            new_path = (audio_folder / metadata.audio_id).with_suffix(".wav")
            # TODO:: process function should also take in sample (AudioClipExample)
            #        and use sample.audio_data when metadata.path does not exist
            if not new_path.exists():
                audio_data = silent_load(
                    str(metadata.path), self.dataset.sr, self.dataset.mono
                )
                soundfile.write(str(new_path), audio_data, self.dataset.sr)
            metadata.path = new_path

        logging.info(f"Writing flat dataset to {folder}...")
        folder.mkdir(exist_ok=True)
        audio_folder = folder / "audio"
        audio_folder.mkdir(exist_ok=True)
        with AudioDatasetMetadataWriter(
            folder, self.dataset.set_type, prefix=self.prefix, mode=self.mode
        ) as writer:
            for metadata in tqdm(
                self.dataset.metadata_list,
                disable=not self.print_progress,
                desc="Writing files",
            ):
                try:
                    process(metadata)
                except EOFError:
                    logging.warning(f"Skipping bad file {metadata.path}")
                    continue
                writer.write(metadata)
