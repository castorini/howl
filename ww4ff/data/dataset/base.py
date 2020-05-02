from dataclasses import dataclass
from typing import Mapping, Any, Optional, List, Tuple
from pathlib import Path
import enum

from pydantic import BaseModel
import torch

from ww4ff.asr import AlignedTranscription


__all__ = ['AudioClipExample',
           'AudioClipMetadata',
           'DatasetType',
           'WakeWordClipExample',
           'ClassificationBatch',
           'AudioDatasetStatistics',
           'EmplacableExample',
           'AlignedAudioClipMetadata',
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


class AlignedAudioClipMetadata(BaseModel):
    path: Path
    transcription: AlignedTranscription


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


@dataclass
class WakeWordClipExample(EmplacableExample):
    metadata: AlignedAudioClipMetadata
    audio_data: torch.Tensor
    sample_rate: int
    frame_labels: Mapping[float, int]

    def emplaced_audio_data(self, audio_data: torch.Tensor) -> 'WakeWordClipExample':
        return WakeWordClipExample(self.metadata, audio_data, self.sample_rate)


class DatasetType(enum.Enum):
    TRAINING = enum.auto()
    DEV = enum.auto()
    TEST = enum.auto()
    UNSPECIFIED = enum.auto()
