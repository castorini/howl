from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, List, Callable

import torch
import torch.utils.data as tud

from .base import DatasetType, AudioClipMetadata, AudioClipExample, WakeWordClipExample, AudioDatasetStatistics, \
    AlignedAudioClipMetadata, ClassificationBatch, NEGATIVE_CLASS, ClassificationClipExample

from ww4ff.settings import SETTINGS
from ww4ff.utils.audio import silent_load


__all__ = ['TypedAudioDataset',
           'AudioClipDataset',
           'WakeWordDataset',
           'AudioClassificationDataset']


class TypedAudioDataset:
    def __init__(self,
                 sr: int = 16000,
                 mono: bool = True,
                 set_type: DatasetType = DatasetType.UNSPECIFIED):
        self.set_type = set_type
        self.sr = sr
        self.mono = mono

    @property
    def is_training(self):
        return self.set_type == DatasetType.TRAINING

    @property
    def is_eval(self):
        return not self.is_training and self.set_type != DatasetType.UNSPECIFIED


class SingleListAttrMixin:
    list_attr = None

    def filter(self, predicate_fn: Callable[[Any], bool], clone: bool = False):
        if clone:
            self = deepcopy(self)
        data_list = self._list_attr
        self._list_attr = list(filter(predicate_fn, data_list))
        return self

    def extend(self, other_dataset: 'SingleListAttrMixin'):
        self._list_attr.extend(other_dataset._list_attr)
        return self

    @property
    def _list_attr(self) -> List[Any]:
        return getattr(self, self.list_attr)

    @_list_attr.setter
    def _list_attr(self, value):
        setattr(self, self.list_attr, value)

    def __len__(self):
        return len(self._list_attr)


class AudioDatasetStatisticsMixin:
    def compute_statistics(self, skip_length: bool = False, use_trim: bool = True) -> AudioDatasetStatistics:
        from ww4ff.data.transform import trim
        seconds = 0
        if not skip_length:
            for ex in self:
                audio_data = trim([ex])[0].audio_data if use_trim else ex.audio_data
                seconds += audio_data.size(-1) / self.sr
        return AudioDatasetStatistics(len(self), seconds)


class AudioClipDataset(AudioDatasetStatisticsMixin, SingleListAttrMixin, TypedAudioDataset, tud.Dataset):
    list_attr = 'metadata_list'

    def __init__(self,
                 metadata_list: List[AudioClipMetadata],
                 **kwargs):
        super().__init__(**kwargs)
        self.metadata_list = metadata_list

    @lru_cache(maxsize=SETTINGS.cache.cache_size)
    def __getitem__(self, idx) -> AudioClipExample:
        metadata = self.metadata_list[idx]
        audio_data = silent_load(str(metadata.path), self.sr, self.mono)
        return AudioClipExample(metadata=metadata, audio_data=torch.from_numpy(audio_data), sample_rate=self.sr)


class WakeWordDataset(AudioDatasetStatisticsMixin, SingleListAttrMixin, TypedAudioDataset, tud.Dataset):
    list_attr = 'metadata_list'

    def __init__(self,
                 metadata_list: List[AlignedAudioClipMetadata],
                 words: List[str],
                 **kwargs):
        super().__init__(**kwargs)
        self.metadata_list = metadata_list
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


class AudioClassificationDataset(AudioDatasetStatisticsMixin, SingleListAttrMixin, TypedAudioDataset, tud.Dataset):
    list_attr = 'metadata_list'

    def __init__(self,
                 metadata_list: List[AudioClipMetadata],
                 label_map: defaultdict,
                 **kwargs):
        super().__init__(**kwargs)
        self.metadata_list = metadata_list
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
