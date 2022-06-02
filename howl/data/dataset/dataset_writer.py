import functools
import multiprocessing
import shutil
from copy import deepcopy
from pathlib import Path

import soundfile
from tqdm import tqdm

from howl.data.common.metadata import AudioClipMetadata
from howl.data.dataset.dataset import AudioClipDataset, DatasetSplit, DatasetType
from howl.dataset.audio_dataset_constants import METADATA_FILE_NAME_TEMPLATES, AudioDatasetType
from howl.dataset.howl_audio_dataset import HowlAudioDataset
from howl.utils.audio_utils import silent_load
from howl.utils.logger import Logger


class AudioDatasetMetadataWriter:
    """Saves audio dataset metadata to the disk"""

    def __init__(self, dataset_path: Path, audio_dataset_type: AudioDatasetType, dataset_split: DatasetSplit):
        """Initialize AudioDatasetMetadataWriter for the given dataset type"""
        self.metadata_json_file = None
        metadata_file_name = METADATA_FILE_NAME_TEMPLATES[audio_dataset_type].format(dataset_split=dataset_split.value)
        self.metadata_json_file_path = str(dataset_path / metadata_file_name)
        self.mode = "w"

    def __enter__(self):
        """Opens the metadata json"""
        self.metadata_json_file = open(self.metadata_json_file_path, self.mode)
        return self

    def write(self, metadata: AudioClipMetadata):
        """Writes metadata to disk"""
        metadata = deepcopy(metadata)
        with metadata.path.with_suffix(".lab").open(self.mode) as metadata_file:
            metadata_file.write(f"{metadata.transcription}\n")
        metadata.path = metadata.path.name
        self.metadata_json_file.write(metadata.json() + "\n")

    def __exit__(self, *args):
        """Closes the metadata json"""
        self.metadata_json_file.close()


class AudioDatasetWriter:
    """Saves audio dataset to the disk"""

    def __init__(self, dataset: AudioClipDataset, audio_dataset_type: AudioDatasetType):
        """Initialize AudioDatasetWriter for the given dataset type"""
        self.dataset = dataset
        self.audio_dataset_type = audio_dataset_type

    @staticmethod
    def _save_audio_file(metadata: AudioClipMetadata, audio_dir_path: Path, sample_rate: int, mono: bool):
        """Generate audio file for the given sample under the folder specified

        Args:
            metadata: metadata for the audio sample
            audio_dir_path: folder of which the audio files will be saved
            sample_rate: sample rate of which the original audio file will be loaded with
            mono: if True, the original audio will be loaded as mono channel

        Returns:
            metadata with updated path
            if audio data cannot be loaded or written correctly, it will return None
        """

        new_audio_file_path = (audio_dir_path / metadata.audio_id).with_suffix(".wav")

        try:
            audio_data = silent_load(str(metadata.path), sample_rate, mono)
            soundfile.write(str(new_audio_file_path), audio_data, sample_rate)
        except Exception as exception:
            Logger.warning(f"Failed to load/write {metadata.path}, the sample will be skipped: {exception}")
            return None

        if not new_audio_file_path.exists():
            shutil.copy(str(metadata.path), str(new_audio_file_path))

        metadata.path = new_audio_file_path
        return metadata

    def write(self, dataset_path: Path):
        """Writes metadata and audio file to disk

        Args:
            dataset_path: dataset path of which the metadata and audio files will be generated
        """

        Logger.info(f"Writing flat dataset to {dataset_path}...")
        dataset_path.mkdir(exist_ok=True)
        audio_dir_path = dataset_path / HowlAudioDataset.DIR_AUDIO
        audio_dir_path.mkdir(exist_ok=True)

        num_processes = max(multiprocessing.cpu_count() // 2, 4)
        pool = multiprocessing.Pool(processes=num_processes)

        metadata_list = tqdm(
            pool.imap(
                functools.partial(
                    AudioDatasetWriter._save_audio_file,
                    audio_dir_path=audio_dir_path,
                    sample_rate=self.dataset.sample_rate,
                    mono=self.dataset.mono,
                ),
                self.dataset.metadata_list,
            ),
            desc=f"Generate {self.dataset.dataset_split} datasets",
            total=(len(self.dataset)),
        )

        self.dataset.metadata_list = list(filter(None, metadata_list))  # remove None entries

        # TODO: to be updated when howl.data.dataset.dataset.DatasetType is replaced
        #       by howl.data.dataset.dataset.DatasetSplit
        if self.dataset.dataset_split == DatasetType.TRAINING:
            dataset_split = DatasetSplit.TRAINING
        elif self.dataset.dataset_split == DatasetType.DEV:
            dataset_split = DatasetSplit.DEV
        elif self.dataset.dataset_split == DatasetType.TEST:
            dataset_split = DatasetSplit.TEST
        else:
            dataset_split = DatasetSplit.UNSPECIFIED

        with AudioDatasetMetadataWriter(dataset_path, self.audio_dataset_type, dataset_split) as metadata_writer:
            for metadata in self.dataset.metadata_list:
                metadata_writer.write(metadata)
