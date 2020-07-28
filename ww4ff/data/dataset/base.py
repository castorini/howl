from dataclasses import dataclass
from typing import Mapping, Optional, List, TypeVar, Generic
from pathlib import Path
import enum

from pydantic import BaseModel
import torch


__all__ = ['AudioClipExample',
           'AudioClipMetadata',
           'AudioDatasetStatistics',
           'ClassificationBatch',
           'ClassificationClipExample',
           'DatasetType',
           'EmplacableExample',
           'AlignedAudioClipMetadata',
           'WakeWordClipExample',
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


class AlignedAudioClipMetadata(AudioClipMetadata):
    end_timestamps: List[float]

    def compute_frame_labels(self, words: List[str]):
        frame_labels = dict()
        t = f' {self.transcription}'
        start = 0
        for idx, word in enumerate(words):
            while True:
                try:
                    start = t.index(word, start)
                except ValueError:
                    break
                frame_labels[self.end_timestamps[start + len(word) - 2]] = idx
                start += 1
        return frame_labels


class EmplacableExample:
    audio_data: torch.Tensor

    def emplaced_audio_data(self,
                            audio_data: torch.Tensor,
                            scale: float = 1,
                            bias: float = 0,
                            new: bool = False) -> 'EmplacableExample':
        raise NotImplementedError


T = TypeVar('T', bound=AudioClipMetadata)


@dataclass
class AudioClipExample(EmplacableExample, Generic[T]):
    metadata: T
    audio_data: torch.Tensor
    sample_rate: int

    def pin_memory(self):
        self.audio_data.pin_memory()

    def emplaced_audio_data(self,
                            audio_data: torch.Tensor,
                            **kwargs) -> 'AudioClipExample':
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
    metadata: T
    audio_data: torch.Tensor
    sample_rate: int
    frame_labels: Mapping[float, int]

    def emplaced_audio_data(self,
                            audio_data: torch.Tensor,
                            scale: float = 1,
                            bias: float = 0,
                            new: bool = False) -> 'WakeWordClipExample':
        frame_labels = {} if new else {scale * k + bias: v for k, v in self.frame_labels.items()}
        return WakeWordClipExample(self.metadata, audio_data, self.sample_rate, frame_labels)


@dataclass
class ClassificationClipExample(EmplacableExample):
    metadata: T
    audio_data: torch.Tensor
    sample_rate: int
    label: int

    def emplaced_audio_data(self, audio_data: torch.Tensor, **kwargs) -> 'ClassificationClipExample':
        return ClassificationClipExample(self.metadata, audio_data, self.sample_rate, self.label)


class DatasetType(enum.Enum):
    TRAINING = enum.auto()
    DEV = enum.auto()
    TEST = enum.auto()
    UNSPECIFIED = enum.auto()
