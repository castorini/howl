import json
import logging
from collections import defaultdict
from copy import deepcopy
from functools import partial
from multiprocessing import Pool
from pathlib import Path
from typing import List, Tuple, TypeVar

import pandas as pd
import soundfile
from tqdm import tqdm

from howl.registered import RegisteredObjectBase
from howl.utils.audio import silent_load
from howl.utils.hash import sha256_int
from howl.utils.transcribe import SpeechToText

from .base import UNKNOWN_TRANSCRIPTION, AudioClipMetadata, DatasetType
from .dataset import (
    AudioClassificationDataset,
    AudioClipDataset,
    AudioDataset,
    WakeWordDataset,
)

__all__ = [
    "AudioDatasetWriter",
    "AudioClipDatasetLoader",
    "MozillaWakeWordLoader",
    "RegisteredPathDatasetLoader",
    "MozillaCommonVoiceLoader",
    "AudioDatasetMetadataWriter",
    "WakeWordDatasetLoader",
    "GoogleSpeechCommandsDatasetLoader",
    "MozillaKeywordLoader",
    "PathDatasetLoader",
    "RecursiveNoiseDatasetLoader",
    "HeySnipsWakeWordLoader",
]


class AudioDatasetMetadataWriter:
    def __init__(self, dataset_path: Path, set_type: DatasetType, prefix: str = "", mode: str = "w"):
        self.filename = str(dataset_path / f"{prefix}metadata-{set_type.name.lower()}.jsonl")
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
    def __init__(self, dataset: AudioClipDataset, prefix: str = "", mode: str = "w", print_progress: bool = True):
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
                audio_data = silent_load(str(metadata.path), self.dataset.sr, self.dataset.mono)
                soundfile.write(str(new_path), audio_data, self.dataset.sr)
            metadata.path = new_path

        logging.info(f"Writing flat dataset to {folder}...")
        folder.mkdir(exist_ok=True)
        audio_folder = folder / "audio"
        audio_folder.mkdir(exist_ok=True)
        with AudioDatasetMetadataWriter(folder, self.dataset.set_type, prefix=self.prefix, mode=self.mode) as writer:
            for metadata in tqdm(self.dataset.metadata_list, disable=not self.print_progress, desc="Writing files"):
                try:
                    process(metadata)
                except EOFError:
                    logging.warning(f"Skipping bad file {metadata.path}")
                    continue
                writer.write(metadata)


class PathDatasetLoader:
    def load_splits(self, path: Path, **dataset_kwargs) -> Tuple[AudioDataset, AudioDataset, AudioDataset]:
        raise NotImplementedError


class RegisteredPathDatasetLoader(PathDatasetLoader, RegisteredObjectBase):
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
        training_path = path / f"{prefix}metadata-{DatasetType.TRAINING.name.lower()}.jsonl"
        dev_path = path / f"{prefix}metadata-{DatasetType.DEV.name.lower()}.jsonl"
        test_path = path / f"{prefix}metadata-{DatasetType.TEST.name.lower()}.jsonl"
        return (
            self.dataset_class(metadata_list=load(training_path), set_type=DatasetType.TRAINING, **dataset_kwargs),
            self.dataset_class(metadata_list=load(dev_path), set_type=DatasetType.DEV, **dataset_kwargs),
            self.dataset_class(metadata_list=load(test_path), set_type=DatasetType.TEST, **dataset_kwargs),
        )


class AudioClipDatasetLoader(MetadataLoaderMixin, RegisteredPathDatasetLoader, name="clip"):
    dataset_class = AudioClipDataset
    metadata_class = AudioClipMetadata


class WakeWordDatasetLoader(MetadataLoaderMixin, PathDatasetLoader):
    default_prefix = "aligned-"
    dataset_class = WakeWordDataset
    metadata_class = AudioClipMetadata


def transcribe_hey_snips_audio(path, metadata):
    stt = SpeechToText()
    path = (path / metadata["audio_file_path"]).absolute()
    transcription = "hey snips"
    if metadata["is_hotword"] == 0:  # negative sample
        transcription = stt.transcribe(path)

    return path, transcription


class HeySnipsWakeWordLoader(RegisteredPathDatasetLoader, name="hey-snips"):
    def __init__(self, num_processes=8):
        self.stt = SpeechToText()
        self.num_processes = num_processes
        self.pool = Pool(processes=self.num_processes)

    def load_splits(self, path: Path, **dataset_kwargs) -> Tuple[AudioClipDataset, AudioClipDataset, AudioClipDataset]:
        self.path = path

        def load(filename, set_type):
            logging.info(f"Loading split {filename}...")

            metadata_list = []
            with open(str(self.path / filename)) as f:
                raw_metadata_list = json.load(f)

            total = len(raw_metadata_list)

            transcription_fail_count = 0
            pbar = tqdm(range(0, total, self.num_processes), desc=f"{filename}")
            for starting_idx in pbar:
                transcription_results = []
                for metadata in raw_metadata_list[starting_idx : starting_idx + self.num_processes]:
                    transcription_results.append(
                        self.pool.apply_async(transcribe_hey_snips_audio, (self.path, metadata,))
                    )

                processing_count = starting_idx + len(transcription_results)

                for result in transcription_results:
                    pbar.set_postfix(dict(transcription_fail_rate=f"{transcription_fail_count}/{processing_count}"))
                    path, transcription = result.get()

                    if transcription == "":
                        transcription_fail_count += 1
                        continue

                    metadata_list.append(AudioClipMetadata(path=path, transcription=transcription))

            logging.info(f"{transcription_fail_count}/{total} samples have empty transcription")

            return AudioClipDataset(metadata_list=metadata_list, set_type=set_type, **dataset_kwargs)

        assert path.exists(), "dataset path doesn't exist"
        filenames = ("train.json", "dev.json", "test.json")
        assert all((path / x).exists() for x in filenames), "dataset missing metadata"

        return (
            load("train.json", DatasetType.TRAINING),
            load("dev.json", DatasetType.DEV),
            load("test.json", DatasetType.TEST),
        )


class GoogleSpeechCommandsDatasetLoader(RegisteredPathDatasetLoader, name="gsc"):
    def __init__(self, vocab: List[str] = None, use_bg_noise: bool = False):
        self.vocab = vocab
        self.use_bg_noise = use_bg_noise

    def load_splits(
        self, path: Path, **dataset_kwargs
    ) -> Tuple[AudioClassificationDataset, AudioClassificationDataset, AudioClassificationDataset]:
        def load(set_type):
            metadata_list = []
            for path in all_list:
                key = str(Path(path.parent.name) / path.name)
                if file_map[key] != set_type:
                    continue
                metadata_list.append(AudioClipMetadata(path=path.absolute(), transcription=path.parent.name))
            return AudioClassificationDataset(
                metadata_list=metadata_list, label_map=label_map, set_type=set_type, **dataset_kwargs
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


class MozillaCommonVoiceLoader(RegisteredPathDatasetLoader, name="mozilla-cv"):
    def load_splits(self, path: Path, **dataset_kwargs) -> Tuple[AudioClipDataset, AudioClipDataset, AudioClipDataset]:
        def load(filename, set_type):
            logging.info(f"Loading split {filename}...")
            df = pd.read_csv(str(path / filename), sep="\t", quoting=3, na_filter=False)
            metadata_list = []
            for tup in df.itertuples():
                metadata_list.append(
                    AudioClipMetadata(path=(path / "clips" / tup.path).absolute(), transcription=tup.sentence)
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
    def load_splits(self, path: Path, **dataset_kwargs) -> Tuple[AudioClipDataset, AudioClipDataset, AudioClipDataset]:
        logging.info("Loading Mozilla keyword dataset...")
        df = pd.read_csv(str(path / "validated.tsv"), sep="\t", quoting=3, na_filter=False)
        md_splits = ([], [], [])
        for tup in df.itertuples():
            md = AudioClipMetadata(path=(path / "clips" / tup.path).absolute(), transcription=tup.sentence)
            bucket = sha256_int(tup.client_id) % 100
            if bucket < 80:
                md_splits[0].append(md)
            elif bucket < 90:
                md_splits[1].append(md)
            else:
                md_splits[2].append(md)
        return (
            AudioClipDataset(metadata_list=md_splits[0], set_type=DatasetType.TRAINING, **dataset_kwargs),
            AudioClipDataset(metadata_list=md_splits[1], set_type=DatasetType.DEV, **dataset_kwargs),
            AudioClipDataset(metadata_list=md_splits[2], set_type=DatasetType.TEST, **dataset_kwargs),
        )


class MozillaWakeWordLoader(RegisteredPathDatasetLoader, name="mozilla-ww"):
    def __init__(self, training_pct=80, dev_pct=10, test_pct=10, split_by_speaker=True, split="verified"):
        self.split_by_speaker = split_by_speaker
        total = training_pct + dev_pct + test_pct
        training_pct = 100 * training_pct / total
        dev_pct = 100 * dev_pct / total
        test_pct = 100 * test_pct / total
        self.cutoffs = (training_pct, dev_pct + training_pct, training_pct + dev_pct + test_pct)
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
                with open(str((sound_fp.parent / sound_id).with_suffix(".txt"))) as f:
                    transcription = f.read()
            else:
                transcription = UNKNOWN_TRANSCRIPTION
            metadata = AudioClipMetadata(path=sound_fp.absolute(), transcription=transcription)
            bucket = sha256_int(speaker_id) if self.split_by_speaker else sha256_int(sound_id)
            bucket %= 100
            bucket = next(idx for idx, cutoff in enumerate(self.cutoffs) if bucket < cutoff)
            metadatas[bucket].append(metadata)
        return (
            AudioClipDataset(metadata_list=metadatas[0], set_type=DatasetType.TRAINING, **dataset_kwargs),
            AudioClipDataset(metadata_list=metadatas[1], set_type=DatasetType.DEV, **dataset_kwargs),
            AudioClipDataset(metadata_list=metadatas[2], set_type=DatasetType.TEST, **dataset_kwargs),
        )


SoundIdSplitMozillaWakeWordLoader = partial(MozillaWakeWordLoader, split_by_speaker=False)
SpeakerSplitMozillaWakeWordLoader = partial(MozillaWakeWordLoader, split_by_speaker=True)


class RecursiveNoiseDatasetLoader:
    def load(self, path: Path, **dataset_kwargs) -> AudioClipDataset:
        wav_names = path.glob("**/*.wav")
        metadata_list = [AudioClipMetadata(path=filename.absolute(), transcription="") for filename in wav_names]
        return AudioClipDataset(metadata_list=metadata_list, set_type=DatasetType.TRAINING, **dataset_kwargs)
