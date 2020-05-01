from typing import Sequence, Iterable
import random

import librosa.effects as effects
import torch
import torch.nn as nn

from ww4ff.data.dataset import WakeWordClipExample, ClassificationBatch, EmplacableExample


__all__ = ['Composition',
           'compose',
           'ZmuvTransform',
           'random_slice',
           'batchify',
           'identity',
           'trim']


class Composition(nn.Module):
    def __init__(self, modules):
        super().__init__()
        self.modules = modules
        self._module_list = nn.ModuleList(list(filter(lambda x: isinstance(x, nn.Module), modules)))

    def forward(self, *args):
        for mod in self.modules:
            args = mod(*args)
            args = (args,)
        return args[0]


class IdentityTransform(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


def compose(*collate_modules):
    return Composition(collate_modules)


def trim(examples: Sequence[EmplacableExample], top_db: int = 40):
    return [ex.emplaced_audio_data(torch.from_numpy(effects.trim(ex.audio_data.cpu().numpy(),
                                                                 top_db=top_db)[0])) for ex in examples]


def random_slice(examples: Sequence[WakeWordClipExample],
                 max_window_size: int = 16000) -> Sequence[WakeWordClipExample]:
    new_examples = []
    for ex in examples:
        if ex.audio_data.size(-1) < max_window_size:
            new_examples.append(ex)
            continue
        a = random.randint(0, ex.audio_data.size(-1) - max_window_size)
        new_examples.append(ex.emplaced_audio_data(ex.audio_data[..., a:a + max_window_size]))
    return new_examples


def batchify(examples: Sequence[WakeWordClipExample]) -> ClassificationBatch:
    examples = sorted(examples, key=lambda x: x.audio_data.size()[-1], reverse=True)
    lengths = torch.tensor([ex.audio_data.size(-1) for ex in examples])
    max_length = max(ex.audio_data.size(-1) for ex in examples)
    audio_tensor = [torch.cat((ex.audio_data.squeeze(), torch.zeros(max_length - ex.audio_data.size(-1))), -1) for
                    ex in examples]
    audio_tensor = torch.stack(audio_tensor)
    labels_tensor = torch.tensor([int(ex.contains_wake_word) for ex in examples])
    return ClassificationBatch(audio_tensor, labels_tensor, lengths)


def identity(x):
    return x


class ZmuvTransform(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('total', torch.zeros(1))
        self.register_buffer('mean', torch.zeros(1))
        self.register_buffer('mean2', torch.zeros(1))

    def update(self, data, mask=None):
        with torch.no_grad():
            if mask is not None:
                data = data * mask
                mask_size = mask.sum().item()
            else:
                mask_size = data.numel()
            self.mean = (data.sum() + self.mean * self.total) / (self.total + mask_size)
            self.mean2 = ((data ** 2).sum() + self.mean2 * self.total) / (self.total + mask_size)
            self.total += mask_size

    def initialize(self, iterable: Iterable[torch.Tensor]):
        for ex in iterable:
            self.update(ex)

    @property
    def std(self):
        return (self.mean2 - self.mean ** 2).sqrt()

    def forward(self, x):
        return (x - self.mean) / self.std
