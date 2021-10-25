import logging
from functools import partial
from pathlib import Path
from typing import Tuple

import pandas as pd

from howl.data.common.metadata import UNKNOWN_TRANSCRIPTION, AudioClipMetadata
from howl.data.dataset.dataset import AudioClipDataset, DatasetType
from howl.data.dataset.dataset_loader import RegisteredPathDatasetLoader
from howl.utils.hash_utils import sha256_int

__all__ = [
    "MozillaCommonVoiceLoader",
    "MozillaKeywordLoader",
    "MozillaWakeWordLoader",
    "SoundIdSplitMozillaWakeWordLoader",
    "SpeakerSplitMozillaWakeWordLoader",
]


class MozillaCommonVoiceLoader(RegisteredPathDatasetLoader, name="mozilla-cv"):
    """DatasetLoader for mozilla common-voice dataset"""

    def load_splits(self, path: Path, **dataset_kwargs) -> Tuple[AudioClipDataset, AudioClipDataset, AudioClipDataset]:
        """Load train/dev/test audio datasets"""

        def load(filename, set_type):
            """Load dataset of given split using the metafile provided"""
            logging.info(f"Loading split {filename}...")
            data_frame = pd.read_csv(str(path / filename), sep="\t", quoting=3, na_filter=False)
            metadata_list = []
            for tup in data_frame.itertuples():
                metadata_list.append(
                    AudioClipMetadata(path=(path / "clips" / tup.path).absolute(), transcription=tup.sentence,)
                )
            return AudioClipDataset(metadata_list=metadata_list, set_type=set_type, **dataset_kwargs)

        assert path.exists(), "dataset path doesn't exist"
        filenames = ("train.tsv", "dev.tsv", "test.tsv")
        assert all((path / x).exists() for x in filenames), "dataset missing metadata"
        return (
            load("train.tsv", DatasetType.TRAINING),
            load("dev.tsv", DatasetType.DEV),
            load("test.tsv", DatasetType.TEST),
        )


class MozillaKeywordLoader(RegisteredPathDatasetLoader, name="mozilla-kw"):
    """DatasetLoader for mozilla keyword dataset"""

    def load_splits(self, path: Path, **dataset_kwargs) -> Tuple[AudioClipDataset, AudioClipDataset, AudioClipDataset]:
        logging.info("Loading Mozilla keyword dataset...")
        data_frame = pd.read_csv(str(path / "validated.tsv"), sep="\t", quoting=3, na_filter=False)
        md_splits = ([], [], [])
        for tup in data_frame.itertuples():
            metadata = AudioClipMetadata(path=(path / "clips" / tup.path).absolute(), transcription=tup.sentence)
            bucket = sha256_int(tup.client_id) % 100
            if bucket < 80:
                md_splits[0].append(metadata)
            elif bucket < 90:
                md_splits[1].append(metadata)
            else:
                md_splits[2].append(metadata)
        return (
            AudioClipDataset(metadata_list=md_splits[0], set_type=DatasetType.TRAINING, **dataset_kwargs,),
            AudioClipDataset(metadata_list=md_splits[1], set_type=DatasetType.DEV, **dataset_kwargs),
            AudioClipDataset(metadata_list=md_splits[2], set_type=DatasetType.TEST, **dataset_kwargs),
        )


class MozillaWakeWordLoader(RegisteredPathDatasetLoader, name="mozilla-ww"):
    """DatasetLoader for mozilla wakeword dataset"""

    def __init__(
        self, training_pct=80, dev_pct=10, test_pct=10, split_by_speaker=True, split="verified",
    ):
        self.split_by_speaker = split_by_speaker
        total = training_pct + dev_pct + test_pct
        training_pct = 100 * training_pct / total
        dev_pct = 100 * dev_pct / total
        test_pct = 100 * test_pct / total
        self.cutoffs = (
            training_pct,
            dev_pct + training_pct,
            training_pct + dev_pct + test_pct,
        )
        self.split = split

    def load_splits(self, path: Path, **dataset_kwargs) -> Tuple[AudioClipDataset, AudioClipDataset, AudioClipDataset]:
        assert path.exists(), "dataset path doesn't exist"
        if self.split in {"verified", "rejected"}:
            audio_path = path / self.split
        else:
            raise ValueError("Split ill-defined.")
        assert audio_path.exists(), "dataset malformed"

        sound_file_paths = audio_path.glob("*/*.ogg")
        metadatas = ([], [], [])
        logging.info("Loading wake word dataset...")
        using_verified = self.split == "verified"
        for sound_fp in sound_file_paths:
            sound_id = sound_fp.stem
            speaker_id = sound_fp.parent.name
            if using_verified:
                with open(str((sound_fp.parent / sound_id).with_suffix(".txt"))) as transcription_file:
                    transcription = transcription_file.read()
            else:
                transcription = UNKNOWN_TRANSCRIPTION
            metadata = AudioClipMetadata(path=sound_fp.absolute(), transcription=transcription)
            bucket = sha256_int(speaker_id) if self.split_by_speaker else sha256_int(sound_id)
            bucket %= 100
            bucket = next(idx for idx, cutoff in enumerate(self.cutoffs) if bucket < cutoff)
            metadatas[bucket].append(metadata)
        return (
            AudioClipDataset(metadata_list=metadatas[0], set_type=DatasetType.TRAINING, **dataset_kwargs,),
            AudioClipDataset(metadata_list=metadatas[1], set_type=DatasetType.DEV, **dataset_kwargs),
            AudioClipDataset(metadata_list=metadatas[2], set_type=DatasetType.TEST, **dataset_kwargs),
        )


SoundIdSplitMozillaWakeWordLoader = partial(MozillaWakeWordLoader, split_by_speaker=False)
SpeakerSplitMozillaWakeWordLoader = partial(MozillaWakeWordLoader, split_by_speaker=True)
