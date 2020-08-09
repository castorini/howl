import torch.nn as nn

from howl.registered import RegisteredObjectBase


__all__ = ['RegisteredModel']


class RegisteredModel(nn.Module, RegisteredObjectBase):
    registered_map = {}

    def __init__(self, num_labels: int):
        super().__init__()
        self.num_labels = num_labels
