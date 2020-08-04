import torch.nn as nn

from howl.registered import RegisteredObjectBase


__all__ = ['RegisteredModel']


class RegisteredModel(nn.Module, RegisteredObjectBase):
    registered_map = {}
