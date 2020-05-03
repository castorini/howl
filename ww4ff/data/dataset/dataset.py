from copy import deepcopy
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, List, Callable

import torch
import torch.utils.data as tud

from .base import DatasetType, AudioClipMetadata, AudioClipExample, WakeWordClipExample, AudioDatasetStatistics, \
    AlignedAudioClipMetadata, ClassificationBatch

from ww4ff.settings import SETTINGS
from ww4ff.utils.audio import silent_load


__all__ = ['TypedAudioDataset', 'AudioClipDataset', 'WakeWordEvaluationDataset', 'WakeWordDataset']


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


class WakeWordEvaluationDataset(TypedAudioDataset, tud.IterableDataset):

    @dataclass
    class Iterator:
        dataset: 'WakeWordEvaluationDataset'
        curr_file_idx = 0
        curr_stride_idx = 0
        label_pointer = None
        frame_labels = None
        output_positives = None

        def __post_init__(self):
            self._init_frame_labels()

        def _init_frame_labels(self):
            self.frame_labels = sorted(list(self.dataset.dataset[self.curr_file_idx].frame_labels.items()),
                                       key=lambda x: x[0])
            self.output_positives = [-1.]
            self.label_pointer = 0

        def _inc(self):
            self.curr_stride_idx = 0
            self.curr_file_idx += 1
            if len(self.dataset.dataset) <= self.curr_file_idx:
                raise StopIteration
            self._init_frame_labels()
            return self.dataset.dataset[self.curr_file_idx]

        def __next__(self) -> ClassificationBatch:
            if len(self.dataset.dataset) <= self.curr_file_idx:
                raise StopIteration
            example = self.dataset.dataset[self.curr_file_idx]
            if example.audio_data.size(-1) < self.dataset.window_size and self.curr_stride_idx != 0:
                example = self._inc()
            elif example.audio_data.size(-1) - self.curr_stride_idx < self.dataset.window_size and \
                    self.curr_stride_idx != 0:
                example = self._inc()
            b = self.curr_stride_idx + self.dataset.window_size
            new_data = example.audio_data[..., self.curr_stride_idx:b]
            end_ts = (b / self.dataset.dataset.sr) * 1000
            try:
                pos_ts, pos_lbl = self.frame_labels[self.label_pointer]
                timedelta = end_ts - pos_ts
                if abs(timedelta) < self.dataset.positive_delta_ms:
                    self.output_positives.append(pos_ts)
                    label = pos_lbl
                else:
                    label = self.dataset.negative_label
                if timedelta > self.dataset.positive_delta_ms:
                    self.label_pointer += 1
                    if self.output_positives[-1] != pos_ts:
                        self.output_positives.append(pos_ts)  # Always output a positive example within an audio clip
                        b = int(pos_ts / 1000 * self.dataset.dataset.sr)
                        a = max(b - self.dataset.window_size, 0)
                        return ClassificationBatch.from_single(example.audio_data[..., a:b], pos_lbl)
            except IndexError:
                label = self.dataset.negative_label
            self.curr_stride_idx += self.dataset.stride_size
            return ClassificationBatch.from_single(new_data, label)

    def __init__(self,
                 wake_word_dataset: WakeWordDataset,
                 window_size: int,
                 stride_size: int,
                 negative_label: int,
                 positive_delta_ms: int = 90):
        super().__init__(sr=wake_word_dataset.sr,
                         mono=wake_word_dataset.mono,
                         set_type=wake_word_dataset.set_type)
        self.window_size = window_size
        self.stride_size = stride_size
        self.positive_delta_ms = positive_delta_ms
        self.dataset = wake_word_dataset
        self.negative_label = negative_label

    def __iter__(self) -> 'WakeWordEvaluationDataset.Iterator':
        return self.Iterator(self)
