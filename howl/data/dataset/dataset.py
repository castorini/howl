import enum
import logging
from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import dataclass
from enum import Enum, unique
from functools import lru_cache
from typing import Any, Callable, Generic, List, TypeVar

import torch
import torch.utils.data as tud

from howl.data.common.example import AudioClipExample, ClassificationClipExample, WakeWordClipExample
from howl.data.common.labeler import FrameLabeler
from howl.data.common.metadata import NEGATIVE_CLASS, AudioClipMetadata
from howl.data.common.searcher import WordTranscriptSearcher
from howl.settings import SETTINGS
from howl.utils.audio import silent_load

__all__ = [
    "DatasetType",
    "AudioDataset",
    "AudioDatasetStatistics",
    "HonkSpeechCommandsDataset",
    "AudioClipDataset",
    "WakeWordDataset",
    "AudioClassificationDataset",
]


@dataclass
class AudioDatasetStatistics:
    num_examples: int
    audio_length_seconds: int
    vocab_counts: Counter


class DatasetType(enum.Enum):
    """To be replaced by SplitType"""

    TRAINING = enum.auto()
    DEV = enum.auto()
    TEST = enum.auto()
    UNSPECIFIED = enum.auto()


@unique
class DatasetSplit(str, Enum):
    """String based Enum of different dataset split"""

    TRAINING = "training"
    DEV = "dev"
    TEST = "test"
    UNSPECIFIED = "unspecified"  # to be removed once DatasetType is fully replaced by DatasetSplit


T = TypeVar("T", bound=AudioClipMetadata)


class AudioDataset(tud.Dataset, Generic[T]):
    def __init__(
        self,
        metadata_list: List[T],
        sr: int = 16000,
        mono: bool = True,
        set_type: DatasetType = DatasetType.UNSPECIFIED,
        split: DatasetSplit = DatasetSplit.UNSPECIFIED,
    ):
        self.metadata_list = metadata_list
        self.set_type = set_type
        self.sr = sr
        self.mono = mono
        self.split = split

    @property
    def is_training(self):
        return self.set_type == DatasetType.TRAINING

    @property
    def is_eval(self):
        return not self.is_training and self.set_type != DatasetType.UNSPECIFIED

    def filter(self, predicate_fn: Callable[[T], bool], clone: bool = False):
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

    def extend(self, other_dataset: "AudioDataset"):
        self.metadata_list.extend(other_dataset.metadata_list)
        return self

    def __len__(self):
        return len(self.metadata_list)

    def compute_statistics(
        self, word_searcher: WordTranscriptSearcher = None, compute_length: bool = True, use_trim: bool = True,
    ) -> AudioDatasetStatistics:
        from howl.data.transform.operator import trim

        seconds = 0
        total_vocab_count = Counter()
        for ex in self:
            if compute_length:
                audio_data = trim([ex])[0].audio_data if use_trim else ex.audio_data
                seconds += audio_data.size(-1) / self.sr
            if word_searcher:
                vocab_count = Counter(word_searcher.count_vocab(ex.metadata.transcription))
                total_vocab_count += vocab_count
        return AudioDatasetStatistics(len(self), seconds, total_vocab_count)

    def print_stats(
        self, logger: logging.Logger = None, header: str = None, **compute_statistics_kwargs,
    ):
        """Print statistics of the dataset

        Args:
            logger: logger to use
            header: additional text message to prepend
            **compute_statistics_kwargs: other arguments passed to compute_statistics
        """
        if header is None:
            log_msg = "Dataset "
        else:
            log_msg = header + " "
        log_msg += f"({self.split.value}) - {self.compute_statistics(**compute_statistics_kwargs)}"

        logger.info(log_msg)


class AudioClipDataset(AudioDataset[AudioClipMetadata]):
    @lru_cache(maxsize=SETTINGS.cache.cache_size)
    def __getitem__(self, idx) -> AudioClipExample:
        metadata = self.metadata_list[idx]
        audio_data = silent_load(str(metadata.path), self.sr, self.mono)
        return AudioClipExample(metadata=metadata, audio_data=torch.from_numpy(audio_data), sample_rate=self.sr,)


class WakeWordDataset(AudioDataset[AudioClipMetadata]):
    def __init__(self, frame_labeler: FrameLabeler, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.frame_labeler = frame_labeler

    @lru_cache(maxsize=SETTINGS.cache.cache_size)
    def __getitem__(self, idx) -> WakeWordClipExample:
        metadata = self.metadata_list[idx]
        audio_data = silent_load(str(metadata.path), self.sr, self.mono)
        return WakeWordClipExample(
            metadata=metadata,
            audio_data=torch.from_numpy(audio_data),
            sample_rate=self.sr,
            label_data=self.frame_labeler.compute_frame_labels(metadata),
        )


class AudioClassificationDataset(AudioDataset[AudioClipMetadata]):
    def __init__(self, label_map: defaultdict, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.label_map = label_map
        self.vocab = {v: k for k, v in label_map.items()}
        self.vocab[label_map.get(None)] = NEGATIVE_CLASS

    @lru_cache(maxsize=SETTINGS.cache.cache_size)
    def __getitem__(self, idx) -> ClassificationClipExample:
        metadata = self.metadata_list[idx]
        audio_data = silent_load(str(metadata.path), self.sr, self.mono)
        return ClassificationClipExample(
            metadata=metadata,
            audio_data=torch.from_numpy(audio_data),
            sample_rate=self.sr,
            label=self.label_map[metadata.transcription],
        )


class HonkSpeechCommandsDataset(AudioClassificationDataset):
    def __init__(self, *args, silence_proportion: float = 0.1, **kwargs):
        super().__init__(*args, **kwargs)
        self.silence_proportion = silence_proportion
        self.silence_label = self.label_map["__silence__"]

    def __getitem__(self, idx) -> ClassificationClipExample:
        if idx < super().__len__():
            return super().__getitem__(idx)
        return ClassificationClipExample(
            metadata=AudioClipMetadata(), audio_data=torch.zeros(16000), sample_rate=self.sr, label=self.silence_label,
        )

    def __len__(self):
        orig_len = super().__len__()
        return orig_len + int(self.silence_proportion * orig_len)
