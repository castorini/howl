from copy import deepcopy
from dataclasses import dataclass
from typing import Mapping, Any, Optional, List, Tuple
from pathlib import Path
import enum

from pydantic import BaseModel
import torch

from ww4ff.align.base import AlignedTranscription


__all__ = ['AudioClipExample',
           'AudioClipMetadata',
           'DatasetType',
           'WakeWordClipExample',
           'ClassificationBatch',
           'AudioDatasetStatistics',
           'EmplacableExample',
           'AlignedAudioClipMetadata',
           'ClassificationClipExample',
           'UNKNOWN_TRANSCRIPTION',
           'NEGATIVE_CLASS']


UNKNOWN_TRANSCRIPTION = '[UNKNOWN]'
NEGATIVE_CLASS = '[NEGATIVE]'


@dataclass
class AudioDatasetStatistics:
    num_examples: int
    audio_length_seconds: int


class AudioClipMetadata(BaseModel):
    path: Path
    transcription: str = ''
    raw: Optional[Mapping[str, Any]]


class AlignedAudioClipMetadata(BaseModel):
    path: Path
    transcription: AlignedTranscription

    def compute_frame_labels(self, words: List[str]):
        frame_labels = dict()
        t = f' {self.transcription.transcription}'
        start = 0
        for idx, word in enumerate(words):
            while True:
                try:
                    start = t.index(word, start)
                except ValueError:
                    break
                frame_labels[self.transcription.end_timestamps[start + len(word) - 2]] = idx
                start += 1
        return frame_labels


class EmplacableExample:
    audio_data: torch.Tensor

    def emplaced_audio_data(self, audio_data: torch.Tensor) -> 'EmplacableExample':
        raise NotImplementedError


@dataclass
class AudioClipExample(EmplacableExample):
    metadata: AudioClipMetadata
    audio_data: torch.Tensor
    sample_rate: int

    def pin_memory(self):
        self.audio_data.pin_memory()

    def emplaced_audio_data(self, audio_data: torch.Tensor) -> 'AudioClipExample':
        return AudioClipExample(self.metadata, audio_data, self.sample_rate)


@dataclass
class ClassificationBatch:
    audio_data: torch.Tensor
    labels: Optional[torch.Tensor]
    lengths: torch.Tensor

    @classmethod
    def from_single(cls, audio_clip: torch.Tensor, label: int) -> 'ClassificationBatch':
        return cls(audio_clip.unsqueeze(0), torch.tensor([label]), torch.tensor([audio_clip.size(-1)]))

    def pin_memory(self):
        self.audio_data.pin_memory()
        self.labels.pin_memory()
        self.lengths.pin_memory()

    def to(self, device: torch.device) -> 'ClassificationBatch':
        self.audio_data = self.audio_data.to(device)
        if self.labels is not None:
            self.labels = self.labels.to(device)
        self.lengths = self.lengths.to(device)
        return self


@dataclass
class WakeWordClipExample(EmplacableExample):
    metadata: AlignedAudioClipMetadata
    audio_data: torch.Tensor
    sample_rate: int
    frame_labels: Mapping[float, int]

    def emplaced_audio_data(self, audio_data: torch.Tensor) -> 'WakeWordClipExample':
        rate = audio_data.size(-1) / self.audio_data.size(-1)
        frame_labels = {rate * k: v for k, v in self.frame_labels.items()}
        return WakeWordClipExample(self.metadata, audio_data, self.sample_rate, frame_labels)


@dataclass
class ClassificationClipExample(EmplacableExample):
    metadata: AudioClipMetadata
    audio_data: torch.Tensor
    sample_rate: int
    label: int

    def emplaced_audio_data(self, audio_data: torch.Tensor) -> 'ClassificationClipExample':
        return ClassificationClipExample(self.metadata, audio_data, self.sample_rate, self.label)


class DatasetType(enum.Enum):
    TRAINING = enum.auto()
    DEV = enum.auto()
    TEST = enum.auto()
    UNSPECIFIED = enum.auto()
