from typing import Any

import torch
import torch.nn as nn

from howl.registered import RegisteredObjectBase


__all__ = ['RegisteredModel', 'ConvertedStaticModel']


class RegisteredModel(nn.Module, RegisteredObjectBase):
    registered_map = {}

    def __init__(self, num_labels: int):
        super().__init__()
        self.num_labels = num_labels
        self.is_streaming = False
        self.is_sequential = False

    def streaming(self):
        self.is_streaming = True
        return self

    def static(self):
        self.is_streaming = False
        return self

    def compute_length(self, length: int):
        return length

    @property
    def streaming_state(self) -> Any:
        return None

    @streaming_state.setter
    def streaming_state(self, x: Any):
        pass


class ConvertedStaticModel(RegisteredModel, name='converted'):
    def __init__(self,
                 model: RegisteredModel,
                 frame_window_size: int,
                 frame_stride_size: int):
        super().__init__(model.num_labels)
        self.model = model
        self.frame_window_size = frame_window_size
        self.frame_stride_size = frame_stride_size

    def compute_length(self, length: int):
        if length is None:
            return None
        return max(1, (length - self.frame_window_size) // self.frame_stride_size)

    def forward(self, x, lengths):
        first = True
        window = x[:, :, :, self.frame_window_size:]
        idx = self.frame_stride_size
        outputs = []
        while first or window.size(3) == self.frame_window_size:
            first = False
            outputs.append(self.model(window, lengths))
            window = x[:, :, :, idx:idx + self.frame_window_size]
            idx += self.frame_stride_size
        return torch.stack(outputs)
