from copy import deepcopy
from dataclasses import dataclass
from typing import Mapping, Optional, List, TypeVar, Generic
from pathlib import Path
import enum

from pydantic import BaseModel
import torch

from .phone import PhonePhrase, Phone


__all__ = ['AudioClipExample',
           'AudioClipMetadata',
           'AudioDatasetStatistics',
           'ClassificationBatch',
           'ClassificationClipExample',
           'DatasetType',
           'EmplacableExample',
           'WakeWordClipExample',
           'FrameLabeler',
           'WordFrameLabeler',
           'FrameLabelData',
           'SequenceBatch',
           'PhoneticFrameLabeler',
           'UNKNOWN_TRANSCRIPTION',
           'NEGATIVE_CLASS']


UNKNOWN_TRANSCRIPTION = '[UNKNOWN]'
NEGATIVE_CLASS = '[NEGATIVE]'


@dataclass
class AudioDatasetStatistics:
    num_examples: int
    audio_length_seconds: int


class AudioClipMetadata(BaseModel):
    path: Optional[Path] = Path('.')
    phone_strings: Optional[List[str]]
    words: Optional[List[str]]
    phone_end_timestamps: Optional[List[float]]
    word_end_timestamps: Optional[List[float]]
    end_timestamps: Optional[List[float]]  # TODO: remove, backwards compat right now
    transcription: Optional[str] = ''

    @property
    def audio_id(self) -> str:
        return self.path.name.split('.', 1)[0]

    @property
    def phone_phrase(self) -> Optional[PhonePhrase]:
        return PhonePhrase([Phone(x) for x in self.phone_strings]) if self.phone_strings else None


@dataclass
class FrameLabelData:
    timestamp_label_map: Mapping[float, int]


class FrameLabeler:
    def compute_frame_labels(self, metadata: AudioClipMetadata) -> FrameLabelData:
        raise NotImplementedError


class PhoneticFrameLabeler(FrameLabeler):
    def __init__(self, phrases: List[PhonePhrase]):
        self.phrases = phrases

    def compute_frame_labels(self, metadata: AudioClipMetadata) -> FrameLabelData:
        frame_labels = dict()
        start = 0
        pp = PhonePhrase.from_string(metadata.transcription)
        for idx, phrase in enumerate(self.phrases):
            while True:
                try:
                    start = pp.audible_index(phrase, start)
                except ValueError:
                    break
                # TODO: make alignment to token instead of character
                start = pp.all_idx_to_transcript_idx(pp.audible_idx_to_all_idx(start))
                frame_labels[metadata.end_timestamps[start + len(str(phrase)) - 1]] = idx
                start += 1
        return FrameLabelData(frame_labels)


class WordFrameLabeler(FrameLabeler):
    def __init__(self, words: List[str], ceil_word_boundary: bool = False):
        self.words = words
        self.ceil_word_boundary = ceil_word_boundary

    def compute_frame_labels(self, metadata: AudioClipMetadata) -> FrameLabelData:
        frame_labels = dict()
        t = f' {metadata.transcription} '
        start = 0
        for idx, word in enumerate(self.words):
            while True:
                try:
                    start = t.index(word, start)
                except ValueError:
                    break
                while self.ceil_word_boundary and start + len(word) < len(t) - 1 and t[start + len(word)] != ' ':
                    start += 1
                frame_labels[metadata.end_timestamps[start + len(word.rstrip()) - 2]] = idx
                start += 1
        return FrameLabelData(frame_labels)


class EmplacableExample:
    audio_data: torch.Tensor

    def emplaced_audio_data(self,
                            audio_data: torch.Tensor,
                            scale: float = 1,
                            bias: float = 0,
                            new: bool = False) -> 'EmplacableExample':
        raise NotImplementedError


T = TypeVar('T', bound=AudioClipMetadata)


class AudioClipExample(EmplacableExample, Generic[T]):
    def __init__(self, metadata: T, audio_data: torch.Tensor, sample_rate: int):
        self.metadata = metadata
        self.audio_data = audio_data
        self.sample_rate = sample_rate

    def pin_memory(self):
        self.audio_data.pin_memory()

    def emplaced_audio_data(self,
                            audio_data: torch.Tensor,
                            scale: float = 1,
                            bias: float = 0,
                            new: bool = False) -> 'AudioClipExample':
        metadata = self.metadata
        if new:
            metadata = deepcopy(metadata)
            metadata.transcription = ''
        return AudioClipExample(metadata, audio_data, self.sample_rate)


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

    def to(self, device: torch.device) -> 'SequenceBatch':
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

    def emplaced_audio_data(self,
                            audio_data: torch.Tensor,
                            scale: float = 1,
                            bias: float = 0,
                            new: bool = False) -> 'WakeWordClipExample':
        ex = super().emplaced_audio_data(audio_data, scale, bias, new)
        label_data = {} if new else {scale * k + bias: v for k, v in self.label_data.timestamp_label_map.items()}
        return WakeWordClipExample(FrameLabelData(label_data), ex.metadata, audio_data, self.sample_rate)


@dataclass
class ClassificationClipExample(AudioClipExample[AudioClipMetadata]):
    def __init__(self, label, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.label = label

    def emplaced_audio_data(self, audio_data: torch.Tensor, **kwargs) -> 'ClassificationClipExample':
        return ClassificationClipExample(self.label, self.metadata, audio_data, self.sample_rate)


class DatasetType(enum.Enum):
    TRAINING = enum.auto()
    DEV = enum.auto()
    TEST = enum.auto()
    UNSPECIFIED = enum.auto()
