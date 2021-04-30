from copy import deepcopy
from dataclasses import dataclass
from typing import Generic, TypeVar

import torch

from howl.core.frame import FrameLabelData
from howl.core.metadata import AudioClipMetadata

__all__ = [
    "AudioClipExample",
    "ClassificationClipExample",
    "EmplacableExample",
    "WakeWordClipExample",
]


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
