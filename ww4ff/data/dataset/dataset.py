from dataclasses import dataclass
from functools import lru_cache
from typing import Any, List, Callable, Iterator, Union

import torch
import torch.utils.data as tud

from .base import DatasetType, AudioClipMetadata, AudioClipExample, WakeWordClipExample, AudioDatasetStatistics
from ww4ff.settings import SETTINGS
from ww4ff.utils.audio import silent_load


__all__ = ['TypedAudioDataset', 'AudioClipDataset', 'WakeWordEvaluationDataset', 'WakeWordTrainingDataset']


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

    def filter(self, predicate_fn: Callable[[Any], bool]):
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

    def compute_statistics(self, skip_length=False) -> AudioDatasetStatistics:
        seconds = 0
        if not skip_length:
            for ex in self:
                seconds += ex.audio_data.size(-1) / self.sr
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


class WakeWordTrainingDataset(AudioDatasetStatisticsMixin, SingleListAttrMixin, TypedAudioDataset, tud.Dataset):
    list_attr = 'metadata_list'

    def __init__(self,
                 audio_clip_dataset: AudioClipDataset,
                 wake_word: str):
        super().__init__(sr=audio_clip_dataset.sr,
                         mono=audio_clip_dataset.mono,
                         set_type=audio_clip_dataset.set_type)
        self.metadata_list = audio_clip_dataset.metadata_list
        self.ds = audio_clip_dataset
        self.wake_word = wake_word

    @lru_cache(maxsize=SETTINGS.cache.cache_size)
    def __getitem__(self, idx) -> WakeWordClipExample:
        ex = self.ds[idx]
        return WakeWordClipExample(metadata=ex.metadata,
                                   audio_data=ex.audio_data,
                                   sample_rate=self.sr,
                                   contains_wake_word=self.wake_word in ex.metadata.transcription)


class WakeWordEvaluationDataset(TypedAudioDataset, tud.IterableDataset):

    @dataclass
    class Iterator:
        dataset: 'WakeWordEvaluationDataset'
        curr_file_idx = 0
        curr_stride_idx = 0

        def _inc(self):
            self.curr_stride_idx = 0
            self.curr_file_idx += 1
            if len(self.dataset.dataset) <= self.curr_file_idx:
                raise StopIteration
            return self.dataset.dataset[self.curr_file_idx]

        def __next__(self) -> WakeWordClipExample:
            if len(self.dataset.dataset) <= self.curr_file_idx:
                raise StopIteration
            example = self.dataset.dataset[self.curr_file_idx]
            if example.audio_data.size(-1) < self.dataset.window_size and self.curr_stride_idx != 0:
                example = self._inc()
            elif example.audio_data.size(-1) - self.curr_stride_idx < self.dataset.window_size and \
                    self.curr_stride_idx != 0:
                example = self._inc()
            new_data = example.audio_data[..., self.curr_stride_idx:self.curr_stride_idx + self.dataset.window_size]
            self.curr_stride_idx += self.dataset.window_size
            return example.emplaced_audio_data(new_data)

    def __init__(self,
                 wake_word_dataset: WakeWordTrainingDataset,
                 window_size: int,
                 stride_size: int):
        super().__init__(sr=wake_word_dataset.sr,
                         mono=wake_word_dataset.mono,
                         set_type=wake_word_dataset.set_type)
        self.window_size = window_size
        self.stride_size = stride_size
        self.dataset = wake_word_dataset

    def __iter__(self) -> 'WakeWordEvaluationDataset.Iterator':
        return self.Iterator(self)
