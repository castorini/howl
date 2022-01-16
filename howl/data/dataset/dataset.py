import enum
import functools
import logging
import multiprocessing
from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import dataclass
from enum import Enum, unique
from functools import lru_cache
from typing import Any, Callable, Generic, List, Tuple, TypeVar

import librosa.effects as effects
import torch
import torch.utils.data as tud
from tqdm import tqdm

from howl.data.common.example import AudioClipExample, ClassificationClipExample, WakeWordClipExample
from howl.data.common.labeler import FrameLabeler
from howl.data.common.metadata import NEGATIVE_CLASS, AudioClipMetadata
from howl.data.common.searcher import WordTranscriptSearcher
from howl.settings import SETTINGS
from howl.utils.audio_utils import silent_load

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
    """Representation of audio dataset statistics"""

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


GenericTypeT = TypeVar("GenericTypeT", bound=AudioClipMetadata)


class AudioDataset(tud.Dataset, Generic[GenericTypeT]):
    """Base representation of an audio dataset"""

    def __init__(
        self,
        metadata_list: List[GenericTypeT],
        sample_rate: int = 16000,
        mono: bool = True,
        set_type: DatasetType = DatasetType.UNSPECIFIED,
        dataset_split: DatasetSplit = DatasetSplit.UNSPECIFIED,
    ):
        """Initialize AudioDataset using the information provided"""
        self.metadata_list = metadata_list
        self.set_type = set_type
        self.sample_rate = sample_rate
        self.mono = mono
        self.dataset_split = dataset_split

    def __repr__(self):
        return repr(f"{self.dataset_split}-dataset")

    @property
    def is_training(self):
        """Return True if the dataset is for training"""
        return self.set_type == DatasetType.TRAINING

    @property
    def is_eval(self):
        """Return True if the dataset is for evaluation"""
        return not self.is_training and self.set_type != DatasetType.UNSPECIFIED

    def filter(self, predicate_fn: Callable[[GenericTypeT], bool], clone: bool = False, **predicate_fn_kwargs):
        """Update current dataset by filtering the relevant samples based on the provided predicate_fn"""
        if clone:
            # TODO: check if deepcopy is necessary
            # pylint: disable=self-cls-assignment
            self = deepcopy(self)
        metadata_list = self.metadata_list
        self.metadata_list = []
        for metadata in tqdm(metadata_list, desc="filtering"):
            if predicate_fn(metadata, **predicate_fn_kwargs):
                self.metadata_list.append(metadata)
        return self

    def split(self, predicate_fn: Callable[[Any], bool]):
        """Split current dataset into two datasets based on the provided predicate_fn"""
        data_list1 = []
        data_list2 = []
        dataset_1 = deepcopy(self)
        dataset_2 = deepcopy(self)
        for metadata in self.metadata_list:
            data_list = data_list2 if predicate_fn(metadata) else data_list1
            data_list.append(metadata)
        dataset_1.metadata_list = data_list1
        dataset_2.metadata_list = data_list2
        return dataset_1, dataset_2

    def extend(self, other_dataset: "AudioDataset"):
        """Combine current dataset with the other dataset"""
        self.metadata_list.extend(other_dataset.metadata_list)
        return self

    def __len__(self):
        """Return number of samples in the dataset"""
        return len(self.metadata_list)

    @staticmethod
    def _compute_sample_statistic(
        metadata: AudioClipMetadata,
        mono: bool,
        sample_rate: int,
        word_searcher: WordTranscriptSearcher,
        compute_length: bool,
        use_trim: bool,
        top_db: int = 40,
    ) -> Tuple[float, Counter]:
        """Compute statistic of the given sample

        Args:
            metadata: metadata of the sample
            mono: if True, the audio file will be loaded assuming the data is mono channel
            sample_rate: sample rate of the audio file
            word_searcher: used to filter out the sample of target vocab
            compute_length: compute total audio length
            use_trim: trim audio data based on decibels before computing total audio length
            top_db: decibels higher than top_db will be trim

        Returns:
            audio data length and vocab counts
        """
        seconds = 0
        vocab_count = Counter()
        if compute_length:
            audio_data = silent_load(str(metadata.path), sample_rate, mono)
            if use_trim:
                audio_data = torch.from_numpy(effects.trim(audio_data, top_db=top_db)[0])
            seconds = audio_data.size(-1) / sample_rate
        if word_searcher:
            vocab_count = Counter(word_searcher.count_vocab(metadata.transcription))
        return seconds, vocab_count

    def compute_statistics(
        self,
        word_searcher: WordTranscriptSearcher = None,
        compute_length: bool = True,
        use_trim: bool = True,
        top_db: int = 40,
    ) -> AudioDatasetStatistics:
        """Compute statistic of the dataset

        Args:
            word_searcher: used to filter out the sample of target vocab
            compute_length: compute total audio length
            use_trim: trim audio data based on decibels before computing total audio length
            top_db: decibels higher than top_db will be trim

        Returns:
            instance of AudioDatasetStatistics

        """
        num_processes = max(multiprocessing.cpu_count() // 2, 4)
        pool = multiprocessing.Pool(processes=num_processes)

        statistics_list = tqdm(
            pool.imap(
                functools.partial(
                    AudioDataset._compute_sample_statistic,
                    mono=self.mono,
                    sample_rate=self.sample_rate,
                    word_searcher=word_searcher,
                    compute_length=compute_length,
                    use_trim=use_trim,
                    top_db=top_db,
                ),
                self.metadata_list,
            ),
            desc="Computing statistics",
            total=(len(self)),
        )

        total_seconds = 0
        total_vocab_count = Counter()
        for statistics in statistics_list:
            total_seconds += statistics[0]
            total_vocab_count += statistics[1]

        return AudioDatasetStatistics(len(self), total_seconds, total_vocab_count)

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
        log_msg += f"({self.dataset_split.value}) - {self.compute_statistics(**compute_statistics_kwargs)}"

        logger.info(log_msg)


# TODO: to be replaced by RawAudioDataset
class AudioClipDataset(AudioDataset[AudioClipMetadata]):
    """A representation of an audio clip dataset"""

    @lru_cache(maxsize=SETTINGS.cache.cache_size)
    def __getitem__(self, idx) -> AudioClipExample:
        """Get sample for the given idx"""
        metadata = self.metadata_list[idx]
        audio_data = silent_load(str(metadata.path), self.sample_rate, self.mono)
        return AudioClipExample(
            metadata=metadata, audio_data=torch.from_numpy(audio_data), sample_rate=self.sample_rate,
        )


class WakeWordDataset(AudioDataset[AudioClipMetadata]):
    """A representation of a wakeword dataset"""

    def __init__(self, frame_labeler: FrameLabeler, *args, **kwargs):
        """Initialize WakeWordDataset"""
        super().__init__(*args, **kwargs)
        self.frame_labeler = frame_labeler

    @lru_cache(maxsize=SETTINGS.cache.cache_size)
    def __getitem__(self, idx) -> WakeWordClipExample:
        """Get sample for the given idx"""
        metadata = self.metadata_list[idx]
        audio_data = silent_load(str(metadata.path), self.sample_rate, self.mono)
        return WakeWordClipExample(
            metadata=metadata,
            audio_data=torch.from_numpy(audio_data),
            sample_rate=self.sample_rate,
            label_data=self.frame_labeler.compute_frame_labels(metadata),
        )


class AudioClassificationDataset(AudioDataset[AudioClipMetadata]):
    """A representation of an audio classification dataset"""

    def __init__(self, label_map: defaultdict, *args, **kwargs):
        """Initialize AudioClassificationDataset"""
        super().__init__(*args, **kwargs)
        self.label_map = label_map
        self.vocab = {v: k for k, v in label_map.items()}
        self.vocab[label_map.get(None)] = NEGATIVE_CLASS

    @lru_cache(maxsize=SETTINGS.cache.cache_size)
    def __getitem__(self, idx) -> ClassificationClipExample:
        """Get sample for the given idx"""
        metadata = self.metadata_list[idx]
        audio_data = silent_load(str(metadata.path), self.sample_rate, self.mono)
        return ClassificationClipExample(
            metadata=metadata,
            audio_data=torch.from_numpy(audio_data),
            sample_rate=self.sample_rate,
            label=self.label_map[metadata.transcription],
        )


class HonkSpeechCommandsDataset(AudioClassificationDataset):
    """A representation of an honk speech commands dataset"""

    def __init__(self, *args, silence_proportion: float = 0.1, **kwargs):
        """Initialize HonkSpeechCommandsDataset"""
        super().__init__(*args, **kwargs)
        self.silence_proportion = silence_proportion
        self.silence_label = self.label_map["__silence__"]

    def __getitem__(self, idx) -> ClassificationClipExample:
        """Get sample for the given idx"""
        if idx < super().__len__():
            return super().__getitem__(idx)
        return ClassificationClipExample(
            metadata=AudioClipMetadata(),
            audio_data=torch.zeros(16000),
            sample_rate=self.sample_rate,
            label=self.silence_label,
        )

    def __len__(self):
        """Get number of samples in the dataset"""
        orig_len = super().__len__()
        return orig_len + int(self.silence_proportion * orig_len)
