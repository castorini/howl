import logging
from copy import deepcopy
from pathlib import Path

import soundfile
from tqdm import tqdm

from howl.data.common.metadata import AudioClipMetadata
from howl.data.dataset.dataset import AudioClipDataset, DatasetType
from howl.utils.audio import silent_load


class AudioDatasetMetadataWriter:
    """Saves audio dataset metadata to the disk"""

    def __init__(
        self, dataset_path: Path, set_type: DatasetType, prefix: str = "", mode: str = "w",
    ):
        """Initialize AudioDatasetMetadataWriter for the given dataset type"""
        self.metadata_json_file = None
        self.metadata_json_file_path = str(dataset_path / f"{prefix}metadata-{set_type.name.lower()}.jsonl")
        self.mode = mode

    def __enter__(self):
        """Opens the metadata json"""
        self.metadata_json_file = open(self.metadata_json_file_path, self.mode)
        return self

    def write(self, metadata: AudioClipMetadata):
        """Writes metadata to disk"""
        metadata = deepcopy(metadata)
        with metadata.path.with_suffix(".lab").open("w") as metadata_file:
            metadata_file.write(f"{metadata.transcription}\n")
        metadata.path = metadata.path.name
        self.metadata_json_file.write(metadata.json() + "\n")

    def __exit__(self, *args):
        """Closes the metadata json"""
        self.metadata_json_file.close()


class AudioDatasetWriter:
    """Saves audio dataset to the disk"""

    def __init__(
        self, dataset: AudioClipDataset, prefix: str = "", mode: str = "w", print_progress: bool = True,
    ):
        """Initialize AudioDatasetWriter for the given dataset type"""
        self.dataset = dataset
        self.print_progress = print_progress
        self.mode = mode
        self.prefix = prefix

    def write(self, folder: Path):
        """Writes metadata and audio file to disk"""

        def process(metadata: AudioClipMetadata):
            """Writes audio file to the path specified in the metadata"""
            new_path = (audio_folder / metadata.audio_id).with_suffix(".wav")
            # TODO:: process function should also take in sample (AudioClipExample)
            #        and use sample.audio_data when metadata.path does not exist
            if not new_path.exists():
                audio_data = silent_load(str(metadata.path), self.dataset.sample_rate, self.dataset.mono)
                soundfile.write(str(new_path), audio_data, self.dataset.sample_rate)
            metadata.path = new_path

        logging.info(f"Writing flat dataset to {folder}...")
        folder.mkdir(exist_ok=True)
        audio_folder = folder / "audio"
        audio_folder.mkdir(exist_ok=True)
        with AudioDatasetMetadataWriter(
            folder, self.dataset.dataset_split, prefix=self.prefix, mode=self.mode
        ) as writer:
            for metadata in tqdm(self.dataset.metadata_list, disable=not self.print_progress, desc="Writing files",):
                try:
                    process(metadata)
                except EOFError as exception:
                    logging.warning(f"Skipping bad file {metadata.path}: {exception}")
                    continue
                writer.write(metadata)
