import torch

from howl.data.common.label import FrameLabelData
from howl.data.common.metadata import AudioClipMetadata


class Sample:
    """Base class for Sample with audio data and corresponding label"""

    def __init__(
        self, metadata: AudioClipMetadata, audio_data: torch.Tensor, sample_rate: int, label: FrameLabelData = None
    ):
        """Represent a sample of audio data

        Args:
            metadata: metadata
            audio_data: audio data
            sample_rate: sample rate of the audio
            label: label data
        """
        self.metadata = metadata
        self.audio_data = audio_data
        self.sample_rate = sample_rate
        self.label = label

    @property
    def labelled(self) -> bool:
        """True if label is available"""
        return self.label is not None

    def pin_memory(self):
        """Pin audio data in memory"""
        self.audio_data.pin_memory()

    def update_data(self, audio_data: torch.Tensor, label: FrameLabelData = None):
        """Update audio data and label

        Args:
            audio_data: new audio data
            label: new label
        """
        self.audio_data = audio_data
        self.label = label
