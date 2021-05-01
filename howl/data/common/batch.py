from dataclasses import dataclass
from typing import Optional

import torch

__all__ = [
    "ClassificationBatch",
    "SequenceBatch",
]


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
