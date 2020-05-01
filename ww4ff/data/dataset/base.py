from dataclasses import dataclass
from typing import Mapping, Any, Optional
from pathlib import Path
import enum

from pydantic import BaseModel
import torch


__all__ = ['AudioClipExample',
           'AudioClipMetadata',
           'DatasetType',
           'WakeWordClipExample',
           'ClassificationBatch',
           'AudioDatasetStatistics',
           'EmplacableExample',
           'UNKNOWN_TRANSCRIPTION']


UNKNOWN_TRANSCRIPTION = '[UNKNOWN]'


@dataclass
class AudioDatasetStatistics:
    num_examples: int
    audio_length_seconds: int


class AudioClipMetadata(BaseModel):
    path: Path
    transcription: str = ''
    raw: Optional[Mapping[str, Any]]


class EmplacableExample:
    audio_data: torch.Tensor

    def emplaced_audio_data(self, audio_data: torch.Tensor) -> 'EmplacableExample':
        raise NotImplementedError


@dataclass(frozen=True)
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
    labels: torch.Tensor
    lengths: torch.Tensor

    def pin_memory(self):
        self.audio_data.pin_memory()
        self.labels.pin_memory()
        self.lengths.pin_memory()

    def to(self, device: torch.device) -> 'ClassificationBatch':
        self.audio_data = self.audio_data.to(device)
        self.labels = self.labels.to(device)
        self.lengths = self.lengths.to(device)
        return self


@dataclass(frozen=True)
class WakeWordClipExample(EmplacableExample):
    metadata: AudioClipMetadata
    audio_data: torch.Tensor
    contains_wake_word: bool
    sample_rate: int

    def emplaced_audio_data(self, audio_data: torch.Tensor) -> 'WakeWordClipExample':
        return WakeWordClipExample(self.metadata, audio_data, self.contains_wake_word, self.sample_rate)


class DatasetType(enum.Enum):
    TRAINING = enum.auto()
    DEV = enum.auto()
    TEST = enum.auto()
    UNSPECIFIED = enum.auto()
