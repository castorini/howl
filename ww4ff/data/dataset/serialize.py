from functools import partial
from typing import Tuple
from pathlib import Path
import json
import logging

from tqdm import tqdm
import pandas as pd
import soundfile

from .base import DatasetType, AudioClipMetadata
from .dataset import AudioClipDataset
from ww4ff.utils.hash import sha256_int


__all__ = ['FlatWavDatasetWriter', 'FlatWavDatasetLoader', 'MozillaWakeWordLoader', 'MozillaCommonVoiceLoader']


class FlatWavDatasetWriter:
    def __init__(self, dataset: AudioClipDataset, print_progress: bool = True):
        self.dataset = dataset
        self.print_progress = print_progress

    def write(self, folder: Path):
        def process(metadata: AudioClipMetadata):
            audio_data = self.dataset.load(metadata.path)
            metadata.path = metadata.path.with_suffix('.wav').name
            soundfile.write(str(metadata.path), audio_data, self.dataset.sr)

        logging.info(f'Writing flat dataset to {folder}...')
        folder.mkdir(exist_ok=True)
        audio_folder = folder / 'audio'
        audio_folder.mkdir(exist_ok=True)
        with open(str(folder / f'metadata-{self.dataset.set_type.name.lower()}.jsonl'), 'w') as f:
            for metadata in tqdm(self.dataset.metadata_list, disable=not self.print_progress, desc='Writing files'):
                process(metadata)
                f.write(metadata.json() + '\n')


class PathDatasetLoader:
    def load_splits(self, path: Path, **dataset_kwargs) -> Tuple[AudioClipDataset, AudioClipDataset, AudioClipDataset]:
        raise NotImplementedError


class FlatWavDatasetLoader(PathDatasetLoader):
    def load_splits(self, path: Path, **dataset_kwargs) -> Tuple[AudioClipDataset, AudioClipDataset, AudioClipDataset]:
        def load(jsonl_name):
            metadata_list = []
            with open(jsonl_name) as f:
                for json_str in iter(f.readline, ''):
                    metadata = AudioClipMetadata(**json.loads(json_str))
                    metadata.path = (path / 'audio' / metadata.path).absolute()
                    metadata_list.append(metadata)
                return metadata_list

        logging.info(f'Loading flat dataset from {path}...')
        training_path = path / f'metadata-{DatasetType.TRAINING.name.lower()}.jsonl'
        dev_path = path / f'metadata-{DatasetType.DEV.name.lower()}.jsonl'
        test_path = path / f'metadata-{DatasetType.TEST.name.lower()}.jsonl'
        return (AudioClipDataset(load(training_path), set_type=DatasetType.TRAINING, **dataset_kwargs),
                AudioClipDataset(load(dev_path), set_type=DatasetType.DEV, **dataset_kwargs),
                AudioClipDataset(load(test_path), set_type=DatasetType.TEST, **dataset_kwargs))


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


class MozillaWakeWordLoader(PathDatasetLoader):
    def __init__(self, training_pct=80, dev_pct=10, test_pct=10, split_by_speaker=False):
        self.split_by_speaker = split_by_speaker
        total = training_pct + dev_pct + test_pct
        training_pct = 100 * training_pct / total
        dev_pct = 100 * dev_pct / total
        test_pct = 100 * test_pct / total
        self.cutoffs = (training_pct, dev_pct + training_pct, training_pct + dev_pct + test_pct)

    def load_splits(self, path: Path, **dataset_kwargs) -> Tuple[AudioClipDataset, AudioClipDataset, AudioClipDataset]:
        assert path.exists(), 'dataset path doesn\'t exist'
        verified_path = path / 'verified'
        assert verified_path.exists(), 'dataset malformed'

        sound_file_paths = verified_path.glob('*/*.ogg')
        metadatas = ([], [], [])
        logging.info('Loading wake word dataset...')
        for sound_fp in sound_file_paths:
            sound_id = sound_fp.stem
            speaker_id = sound_fp.parent.name
            with open(str((sound_fp.parent / sound_id).with_suffix('.txt'))) as f:
                transcription = f.read()
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
