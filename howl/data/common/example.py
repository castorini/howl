from copy import deepcopy
from dataclasses import dataclass
from typing import Generic, TypeVar

import torch

from howl.data.common.label import FrameLabelData
from howl.data.common.metadata import AudioClipMetadata

__all__ = [
    "AudioClipExample",
    "ClassificationClipExample",
    "WakeWordClipExample",
]


class EmplacableExample:
    """Interface class for sample with audio data which can be modified"""

    audio_data: torch.Tensor

    def update_audio_data(
        self, audio_data: torch.Tensor, scale: float = 1, bias: float = 0, new: bool = False
    ) -> "EmplacableExample":
        """Abstract method for updating audio data

        Args:
            audio_data: new audio data
            scale: ratio of which the audio data needs to be stretched (Optional)
            bias: bias value that needs to be added to audio data (Optional)
            new: if True, new instance of EmplacableExample will be created

        Returns:
            EmplacableExample abject with the updated audio data
        """
        raise NotImplementedError


Metadata = TypeVar("Metadata", bound=AudioClipMetadata)


class AudioClipExample(EmplacableExample, Generic[Metadata]):
    """A sample consisting audio data without label"""

    def __init__(self, metadata: Metadata, audio_data: torch.Tensor, sample_rate: int):
        self.metadata = metadata
        self.audio_data = audio_data
        self.sample_rate = sample_rate

    def pin_memory(self):
        """Pin audio data in memory"""
        self.audio_data.pin_memory()

    def update_audio_data(
        self, audio_data: torch.Tensor, scale: float = 1, bias: float = 0, new: bool = False
    ) -> "AudioClipExample":
        """Abstract method for updating audio data

        Args:
            audio_data: new audio data
            scale: n/a
            bias: n/a
            new: if True, new instance of AudioClipExample will be created

        Returns:
            AudioClipExample abject with the updated audio data
        """
        metadata = self.metadata
        if new:
            metadata = deepcopy(metadata)
            metadata.transcription = ""
        return AudioClipExample(metadata, audio_data, self.sample_rate)


@dataclass
class WakeWordClipExample(AudioClipExample[AudioClipMetadata]):
    """A sample consisting audio data with FrameLabelData"""

    def __init__(self, label_data: FrameLabelData, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.label_data = label_data

    def update_audio_data(
        self, audio_data: torch.Tensor, scale: float = 1, bias: float = 0, new: bool = False
    ) -> "WakeWordClipExample":
        """Update the label based on the provided audio data augmentation and create new instance of WakeWordClipExample

        Args:
            audio_data: new audio data
            scale: ratio of which the audio data needs to be stretched (Optional)
            bias: bias value that needs to be added to audio data (Optional)
            new: n/a

        Returns:
            WakeWordClipExample abject with the updated audio data and label data
        """
        ex = super().update_audio_data(audio_data, scale, bias, new)
        label_data = {} if new else {scale * k + bias: v for k, v in self.label_data.timestamp_label_map.items()}
        return WakeWordClipExample(
            FrameLabelData(label_data, self.label_data.start_timestamp, self.label_data.char_indices),
            ex.metadata,
            audio_data,
            self.sample_rate,
        )


# TODO: check how WakeWordClipExample is differ from ClassificationClipExample and combine them if possible
@dataclass
class ClassificationClipExample(AudioClipExample[AudioClipMetadata]):
    """A sample consisting audio data with generic label"""

    def __init__(self, label, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.label = label

    def update_audio_data(self, audio_data: torch.Tensor, **kwargs) -> "ClassificationClipExample":
        """Creates new ClassificationClipExample instance with the provided audio data

        Args:
            audio_data: audio data of which the new ClassificationClipExample will be created with
            kwargs: n/a

        Returns:
            ClassificationClipExample instance with new audio data
        """
        # pylint: disable=unused-argument
        # pylint: disable=arguments-differ
        return ClassificationClipExample(self.label, self.metadata, audio_data, self.sample_rate)
