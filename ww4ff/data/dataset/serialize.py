from collections import defaultdict
from copy import deepcopy
from functools import partial
from typing import Tuple
from pathlib import Path
import json
import logging

from pydantic import BaseModel
from tqdm import tqdm
import pandas as pd
import soundfile

from .base import DatasetType, AudioClipMetadata, UNKNOWN_TRANSCRIPTION, AlignedAudioClipMetadata
from .dataset import AudioClipDataset, WakeWordDataset, AudioClassificationDataset
from ww4ff.utils.audio import silent_load
from ww4ff.utils.hash import sha256_int


__all__ = ['AudioClipDatasetWriter',
           'AudioClipDatasetLoader',
           'MozillaWakeWordLoader',
           'MozillaCommonVoiceLoader',
           'AudioClipDatasetMetadataWriter',
           'WakeWordDatasetLoader',
           'GoogleSpeechCommandsDatasetLoader',
           'MozillaKeywordLoader']


class AudioClipDatasetMetadataWriter:
    def __init__(self, dataset_path: Path, set_type: DatasetType, prefix: str = '', mode: str = 'w'):
        self.filename = str(dataset_path / f'{prefix}metadata-{set_type.name.lower()}.jsonl')
        self.mode = mode

    def __enter__(self):
        self.f = open(self.filename, self.mode)
        return self

    def write(self, metadata: BaseModel):
        metadata = deepcopy(metadata)
        metadata.path = metadata.path.name
        self.f.write(metadata.json() + '\n')

    def __exit__(self, *args):
        self.f.close()


class AudioClipDatasetWriter:
    def __init__(self, dataset: AudioClipDataset, mode: str = 'w', print_progress: bool = True):
        self.dataset = dataset
        self.print_progress = print_progress
        self.mode = mode

    def write(self, folder: Path):
        def process(metadata: AudioClipMetadata):
            audio_data = silent_load(str(metadata.path), self.dataset.sr, self.dataset.mono)
            metadata.path = audio_folder / metadata.path.with_suffix('.wav').name
            soundfile.write(str(metadata.path), audio_data, self.dataset.sr)

        logging.info(f'Writing flat dataset to {folder}...')
        folder.mkdir(exist_ok=True)
        audio_folder = folder / 'audio'
        audio_folder.mkdir(exist_ok=True)
        with AudioClipDatasetMetadataWriter(audio_folder, self.dataset.set_type, mode=self.mode) as writer:
            for metadata in tqdm(self.dataset.metadata_list, disable=not self.print_progress, desc='Writing files'):
                try:
                    process(metadata)
                except EOFError:
                    logging.warning(f'Skipping bad file {metadata.path}')
                    continue
                writer.write(metadata)


class PathDatasetLoader:
    def load_splits(self, path: Path, **dataset_kwargs) -> Tuple[AudioClipDataset, AudioClipDataset, AudioClipDataset]:
        raise NotImplementedError


class MetadataLoaderMixin:
    dataset_class = None
    metadata_class = None
    default_prefix = ''

    def load_splits(self,
                    path: Path,
                    prefix: str = None,
                    **dataset_kwargs):
        def load(jsonl_name):
            metadata_list = []
            with open(jsonl_name) as f:
                for json_str in iter(f.readline, ''):
                    metadata = self.metadata_class(**json.loads(json_str))
                    metadata.path = (path / 'audio' / metadata.path).absolute()
                    metadata_list.append(metadata)
                return metadata_list

        if prefix is None:
            prefix = self.default_prefix
        logging.info(f'Loading flat dataset from {path}...')
        training_path = path / f'{prefix}metadata-{DatasetType.TRAINING.name.lower()}.jsonl'
        dev_path = path / f'{prefix}metadata-{DatasetType.DEV.name.lower()}.jsonl'
        test_path = path / f'{prefix}metadata-{DatasetType.TEST.name.lower()}.jsonl'
        return (self.dataset_class(load(training_path), set_type=DatasetType.TRAINING, **dataset_kwargs),
                self.dataset_class(load(dev_path), set_type=DatasetType.DEV, **dataset_kwargs),
                self.dataset_class(load(test_path), set_type=DatasetType.TEST, **dataset_kwargs))


class AudioClipDatasetLoader(MetadataLoaderMixin, PathDatasetLoader):
    dataset_class = AudioClipDataset
    metadata_class = AudioClipMetadata


class WakeWordDatasetLoader(MetadataLoaderMixin, PathDatasetLoader):
    default_prefix = 'aligned-'
    dataset_class = WakeWordDataset
    metadata_class = AlignedAudioClipMetadata


class GoogleSpeechCommandsDatasetLoader(PathDatasetLoader):
    def load_splits(self, path: Path, **dataset_kwargs) -> Tuple[AudioClassificationDataset,
                                                                 AudioClassificationDataset,
                                                                 AudioClassificationDataset]:
        def load(set_type):
            metadata_list = []
            for path in all_list:
                key = str(Path(path.parent.name) / path.name)
                if file_map[key] != set_type:
                    continue
                metadata_list.append(AudioClipMetadata(path=path.absolute(),
                                                       transcription=path.parent.name))
            return AudioClassificationDataset(metadata_list,
                                              label_map,
                                              set_type=set_type,
                                              **dataset_kwargs)

        file_map = defaultdict(lambda: DatasetType.TRAINING)
        with open(path/ 'testing_list.txt') as f:
            file_map.update({k: DatasetType.TEST for k in f.read().split('\n')})
        with open(path / 'validation_list.txt') as f:
            file_map.update({k: DatasetType.DEV for k in f.read().split('\n')})
        all_list = list(path.glob('*/*.wav'))
        folders = sorted(list(path.glob('*/')))
        label_map = defaultdict(lambda: len(folders))
        label_map.update({k.name: idx for idx, k in enumerate(folders)})
        return load(DatasetType.TRAINING), load(DatasetType.DEV), load(DatasetType.TEST)


class MozillaCommonVoiceLoader(PathDatasetLoader):
    def load_splits(self, path: Path, **dataset_kwargs) -> Tuple[AudioClipDataset, AudioClipDataset, AudioClipDataset]:
        def load(filename, set_type):
            logging.info(f'Loading split {filename}...')
            df = pd.read_csv(str(path / filename), sep='\t', quoting=3, na_filter=False)
            metadata_list = []
            for tup in df.itertuples():
                metadata_list.append(AudioClipMetadata(path=(path / 'clips' / tup.path).absolute(), transcription=tup.sentence))
            return AudioClipDataset(metadata_list, set_type=set_type, **dataset_kwargs)

        assert path.exists(), 'dataset path doesn\'t exist'
        filenames = ('train.tsv', 'dev.tsv', 'test.tsv')
        assert all((path / x).exists() for x in filenames), 'dataset missing metadata'
        return (load('train.tsv', DatasetType.TRAINING),
                load('dev.tsv', DatasetType.DEV),
                load('test.tsv', DatasetType.TEST))


class MozillaKeywordLoader(PathDatasetLoader):
    def load_splits(self, path: Path, **dataset_kwargs) -> Tuple[AudioClipDataset, AudioClipDataset, AudioClipDataset]:
        logging.info(f'Loading Mozilla keyword dataset...')
        df = pd.read_csv(str(path / 'validated.tsv'), sep='\t', quoting=3, na_filter=False)
        md_splits = ([], [], [])
        for tup in df.itertuples():
            md = AudioClipMetadata(path=(path / 'clips' / tup.path).absolute(), transcription=tup.sentence)
            bucket = sha256_int(tup.client_id) % 100
            if bucket < 80:
                md_splits[0].append(md)
            elif bucket < 90:
                md_splits[1].append(md)
            else:
                md_splits[2].append(md)
        return (AudioClipDataset(md_splits[0], set_type=DatasetType.TRAINING, **dataset_kwargs),
                AudioClipDataset(md_splits[1], set_type=DatasetType.DEV, **dataset_kwargs),
                AudioClipDataset(md_splits[2], set_type=DatasetType.TEST, **dataset_kwargs))


class MozillaWakeWordLoader(PathDatasetLoader):
    def __init__(self, training_pct=80, dev_pct=10, test_pct=10, split_by_speaker=False, split='verified'):
        self.split_by_speaker = split_by_speaker
        total = training_pct + dev_pct + test_pct
        training_pct = 100 * training_pct / total
        dev_pct = 100 * dev_pct / total
        test_pct = 100 * test_pct / total
        self.cutoffs = (training_pct, dev_pct + training_pct, training_pct + dev_pct + test_pct)
        self.split = split

    def load_splits(self, path: Path, **dataset_kwargs) -> Tuple[AudioClipDataset, AudioClipDataset, AudioClipDataset]:
        assert path.exists(), 'dataset path doesn\'t exist'
        if self.split in {'verified', 'rejected'}:
            audio_path = path / self.split
        else:
            raise ValueError('Split ill-defined.')
        assert audio_path.exists(), 'dataset malformed'

        sound_file_paths = audio_path.glob('*/*.ogg')
        metadatas = ([], [], [])
        logging.info('Loading wake word dataset...')
        using_verified = self.split == 'verified'
        for sound_fp in sound_file_paths:
            sound_id = sound_fp.stem
            speaker_id = sound_fp.parent.name
            if using_verified:
                with open(str((sound_fp.parent / sound_id).with_suffix('.txt'))) as f:
                    transcription = f.read()
            else:
                transcription = UNKNOWN_TRANSCRIPTION
            metadata = AudioClipMetadata(path=sound_fp.absolute(), transcription=transcription)
            bucket = sha256_int(speaker_id) if self.split_by_speaker else sha256_int(sound_id)
            bucket %= 100
            bucket = next(idx for idx, cutoff in enumerate(self.cutoffs) if bucket < cutoff)
            metadatas[bucket].append(metadata)
        return (AudioClipDataset(metadatas[0], set_type=DatasetType.TRAINING, **dataset_kwargs),
                AudioClipDataset(metadatas[1], set_type=DatasetType.DEV, **dataset_kwargs),
                AudioClipDataset(metadatas[2], set_type=DatasetType.TEST, **dataset_kwargs))


SoundIdSplitMozillaWakeWordLoader = partial(MozillaWakeWordLoader, split_by_speaker=False)
SpeakerSplitMozillaWakeWordLoader = partial(MozillaWakeWordLoader, split_by_speaker=True)
