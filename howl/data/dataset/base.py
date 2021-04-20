import enum
from collections import Counter
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Generic, List, Mapping, Optional, Tuple, TypeVar

import torch
from pydantic import BaseModel

from .phone import Phone, PhonePhrase

__all__ = [
    "AudioClipExample",
    "AudioClipMetadata",
    "AudioDatasetStatistics",
    "ClassificationBatch",
    "ClassificationClipExample",
    "DatasetType",
    "EmplacableExample",
    "WakeWordClipExample",
    "SequenceBatch",
    "FrameLabelData",
    "UNKNOWN_TRANSCRIPTION",
    "NEGATIVE_CLASS",
]


UNKNOWN_TRANSCRIPTION = "[UNKNOWN]"
NEGATIVE_CLASS = "[NEGATIVE]"


@dataclass
class FrameLabelData:
    timestamp_label_map: Mapping[float, int]
    start_timestamp: List[Tuple[int, float]]
    char_indices: List[Tuple[int, List[int]]]


@dataclass
class AudioDatasetStatistics:
    num_examples: int
    audio_length_seconds: int
    vocab_counts: Counter


class AudioClipMetadata(BaseModel):
    path: Optional[Path] = Path(".")
    phone_strings: Optional[List[str]]
    words: Optional[List[str]]
    phone_end_timestamps: Optional[List[float]]
    word_end_timestamps: Optional[List[float]]
    end_timestamps: Optional[List[float]]  # TODO: remove, backwards compat right now
    transcription: Optional[str] = ""

    # TODO:: id should be an explicit variable in order to support datasets creation with the audio data in memory
    @property
    def audio_id(self) -> str:
        return self.path.name.split(".", 1)[0]

    @property
    def phone_phrase(self) -> Optional[PhonePhrase]:
        return PhonePhrase([Phone(x) for x in self.phone_strings]) if self.phone_strings else None


class EmplacableExample:
    audio_data: torch.Tensor

    def emplaced_audio_data(
        self, audio_data: torch.Tensor, scale: float = 1, bias: float = 0, new: bool = False
    ) -> "EmplacableExample":
        raise NotImplementedError


T = TypeVar("T", bound=AudioClipMetadata)


class AudioClipExample(EmplacableExample, Generic[T]):
    def __init__(self, metadata: T, audio_data: torch.Tensor, sample_rate: int):
        self.metadata = metadata
        self.audio_data = audio_data
        self.sample_rate = sample_rate

    def pin_memory(self):
        self.audio_data.pin_memory()

    def emplaced_audio_data(
        self, audio_data: torch.Tensor, scale: float = 1, bias: float = 0, new: bool = False
    ) -> "AudioClipExample":
        metadata = self.metadata
        if new:
            metadata = deepcopy(metadata)
            metadata.transcription = ""
        return AudioClipExample(metadata, audio_data, self.sample_rate)


@dataclass
class ClassificationBatch:
    audio_data: torch.Tensor
    labels: Optional[torch.Tensor]
    lengths: torch.Tensor

    @classmethod
    def from_single(cls, audio_clip: torch.Tensor, label: int) -> "ClassificationBatch":
        return cls(audio_clip.unsqueeze(0), torch.tensor([label]), torch.tensor([audio_clip.size(-1)]))

    def pin_memory(self):
        self.audio_data.pin_memory()
        self.labels.pin_memory()
        self.lengths.pin_memory()

    def to(self, device: torch.device) -> "ClassificationBatch":
        self.audio_data = self.audio_data.to(device)
        if self.labels is not None:
            self.labels = self.labels.to(device)
        self.lengths = self.lengths.to(device)
        return self


@dataclass
class SequenceBatch:
    audio_data: torch.Tensor  # (batch size, max audio length)
    labels: torch.Tensor  # (batch size, max label length)
    audio_lengths: Optional[torch.Tensor]
    label_lengths: Optional[torch.Tensor]

    def __post_init__(self):
        if self.audio_lengths is None:
            self.audio_lengths = torch.tensor([self.audio_data.size(0) for _ in range(self.audio_data.size(1))])
            self.audio_lengths = self.audio_lengths.to(self.audio_data.device)
        if self.label_lengths is None:
            self.label_lengths = torch.tensor([self.labels.size(0) for _ in range(self.labels.size(1))])
            self.label_lengths = self.label_lengths.to(self.label_lengths.device)

    def pin_memory(self):
        self.audio_data.pin_memory()
        self.labels.pin_memory()
        self.audio_lengths.pin_memory()
        self.label_lengths.pin_memory()

    def to(self, device: torch.device) -> "SequenceBatch":
        self.audio_data = self.audio_data.to(device)
        self.labels = self.labels.to(device)
        self.audio_lengths = self.audio_lengths.to(device)
        self.label_lengths = self.label_lengths.to(device)
        return self


@dataclass
class WakeWordClipExample(AudioClipExample[AudioClipMetadata]):
    def __init__(self, label_data: FrameLabelData, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.label_data = label_data

    def emplaced_audio_data(
        self, audio_data: torch.Tensor, scale: float = 1, bias: float = 0, new: bool = False
    ) -> "WakeWordClipExample":
        ex = super().emplaced_audio_data(audio_data, scale, bias, new)
        label_data = {} if new else {scale * k + bias: v for k, v in self.label_data.timestamp_label_map.items()}
        return WakeWordClipExample(
            FrameLabelData(label_data, self.label_data.start_timestamp, self.label_data.char_indices),
            ex.metadata,
            audio_data,
            self.sample_rate,
        )


@dataclass
class ClassificationClipExample(AudioClipExample[AudioClipMetadata]):
    def __init__(self, label, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.label = label

    def emplaced_audio_data(self, audio_data: torch.Tensor, **kwargs) -> "ClassificationClipExample":
        return ClassificationClipExample(self.label, self.metadata, audio_data, self.sample_rate)


class DatasetType(enum.Enum):
    TRAINING = enum.auto()
    DEV = enum.auto()
    TEST = enum.auto()
    UNSPECIFIED = enum.auto()
