from collections import defaultdict
from copy import deepcopy
from functools import lru_cache
from typing import Any, List, Callable, TypeVar, Generic

import torch
import torch.utils.data as tud

from .base import DatasetType, AudioClipMetadata, AudioClipExample, WakeWordClipExample, AudioDatasetStatistics, \
    AlignedAudioClipMetadata, NEGATIVE_CLASS, ClassificationClipExample
from ww4ff.settings import SETTINGS
from ww4ff.utils.audio import silent_load
from ww4ff.utils.hash import sha256_int


__all__ = ['AudioDataset',
           'AudioClipDataset',
           'WakeWordDataset',
           'AudioClassificationDataset',
           'Sha256Splitter']


T = TypeVar('T', bound=AudioClipMetadata)


class AudioDataset(tud.Dataset, Generic[T]):
    def __init__(self,
                 metadata_list: List[T],
                 sr: int = 16000,
                 mono: bool = True,
                 set_type: DatasetType = DatasetType.UNSPECIFIED):
        self.metadata_list = metadata_list
        self.set_type = set_type
        self.sr = sr
        self.mono = mono

    @property
    def is_training(self):
        return self.set_type == DatasetType.TRAINING

    @property
    def is_eval(self):
        return not self.is_training and self.set_type != DatasetType.UNSPECIFIED

    def filter(self, predicate_fn: Callable[[Any], bool], clone: bool = False):
        if clone:
            self = deepcopy(self)
        data_list = self.metadata_list
        self.metadata_list = list(filter(predicate_fn, data_list))
        return self

    def split(self, predicate_fn: Callable[[Any], bool]):
        data_list1 = []
        data_list2 = []
        x1 = deepcopy(self)
        x2 = deepcopy(self)
        for x in self.metadata_list:
            data_list = data_list2 if predicate_fn(x) else data_list1
            data_list.append(x)
        x1.metadata_list = data_list1
        x2.metadata_list = data_list2
        return x1, x2

    def extend(self, other_dataset: 'AudioDataset'):
        self.metadata_list.extend(other_dataset.metadata_list)
        return self

    def __len__(self):
        return len(self.metadata_list)

    def compute_statistics(self, skip_length: bool = False, use_trim: bool = True) -> AudioDatasetStatistics:
        from ww4ff.data.transform import trim
        seconds = 0
        if not skip_length:
            for ex in self:
                audio_data = trim([ex])[0].audio_data if use_trim else ex.audio_data
                seconds += audio_data.size(-1) / self.sr
        return AudioDatasetStatistics(len(self), seconds)


class AudioClipDataset(AudioDataset[AudioClipMetadata]):
    @lru_cache(maxsize=SETTINGS.cache.cache_size)
    def __getitem__(self, idx) -> AudioClipExample:
        metadata = self.metadata_list[idx]
        audio_data = silent_load(str(metadata.path), self.sr, self.mono)
        return AudioClipExample(metadata=metadata, audio_data=torch.from_numpy(audio_data), sample_rate=self.sr)


class WakeWordDataset(AudioDataset[AlignedAudioClipMetadata]):
    def __init__(self,
                 words: List[str],
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.words = words
        self.vocab = {v: k for k, v in enumerate(words)}

    @lru_cache(maxsize=SETTINGS.cache.cache_size)
    def __getitem__(self, idx) -> WakeWordClipExample:
        metadata = self.metadata_list[idx]
        audio_data = silent_load(str(metadata.path), self.sr, self.mono)
        frame_labels = metadata.compute_frame_labels(self.words)
        return WakeWordClipExample(metadata=metadata,
                                   audio_data=torch.from_numpy(audio_data),
                                   sample_rate=self.sr,
                                   frame_labels=frame_labels)


class AudioClassificationDataset(AudioDataset[AudioClipMetadata]):
    def __init__(self,
                 label_map: defaultdict,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.label_map = label_map
        self.vocab = {v: k for k, v in label_map.items()}
        self.vocab[label_map.get(None)] = NEGATIVE_CLASS

    @lru_cache(maxsize=SETTINGS.cache.cache_size)
    def __getitem__(self, idx) -> ClassificationClipExample:
        metadata = self.metadata_list[idx]
        audio_data = silent_load(str(metadata.path), self.sr, self.mono)
        return ClassificationClipExample(metadata=metadata,
                                         audio_data=torch.from_numpy(audio_data),
                                         sample_rate=self.sr,
                                         label=self.label_map[metadata.transcription])


class Sha256Splitter:
    def __init__(self, target_pct: int):
        self.target_pct = target_pct

    def __call__(self, x: AudioClipMetadata) -> bool:
        return (sha256_int(str(x.path)) % 100) < self.target_pct
