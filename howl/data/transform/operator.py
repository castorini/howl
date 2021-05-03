import random
from typing import Iterable, List, Sequence

import librosa.effects as effects
import numpy as np
import torch
import torch.nn as nn

from howl.data.common.batch import ClassificationBatch
from howl.data.common.example import EmplacableExample, WakeWordClipExample

__all__ = [
    "Composition",
    "compose",
    "ZmuvTransform",
    "random_slice",
    "batchify",
    "identity",
    "trim",
    "truncate_length",
]


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


def compose(*collate_modules):
    return Composition(collate_modules)


class IdentityTransform(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


def identity(x):
    return x


def trim(examples: Sequence[EmplacableExample], top_db: int = 40):
    return [
        ex.emplaced_audio_data(torch.from_numpy(effects.trim(ex.audio_data.cpu().numpy(), top_db=top_db)[0]))
        for ex in examples
    ]


def random_slice(
    examples: Sequence[WakeWordClipExample], max_window_size: int = 16000
) -> Sequence[WakeWordClipExample]:
    new_examples = []
    for ex in examples:
        if ex.audio_data.size(-1) < max_window_size:
            new_examples.append(ex)
            continue
        a = random.randint(0, ex.audio_data.size(-1) - max_window_size)
        new_examples.append(ex.emplaced_audio_data(ex.audio_data[..., a : a + max_window_size]))
    return new_examples


def truncate_length(examples: Sequence[EmplacableExample], length: int = None):
    return [ex.emplaced_audio_data(ex.audio_data[..., :length]) for ex in examples]


def batchify(examples: Sequence[EmplacableExample], label_provider=None):
    examples = sorted(examples, key=lambda x: x.audio_data.size()[-1], reverse=True)
    lengths = torch.tensor([ex.audio_data.size(-1) for ex in examples])
    max_length = max(ex.audio_data.size(-1) for ex in examples)
    audio_tensor = [
        torch.cat((ex.audio_data.squeeze(), torch.zeros(max_length - ex.audio_data.size(-1))), -1) for ex in examples
    ]
    audio_tensor = torch.stack(audio_tensor)
    labels = torch.tensor(list(map(label_provider, examples))) if label_provider else None
    return ClassificationBatch(audio_tensor, labels, lengths)


def tensorize_audio_data(
    audio_data_lst: List[torch.Tensor], max_length: int = None, rand_append: bool = False, **extra_data_lists
):
    lengths = np.array([audio_data.size(-1) for audio_data in audio_data_lst])
    sort_indices = np.argsort(-lengths)
    audio_data_lst = np.array(audio_data_lst, dtype=object)[sort_indices].tolist()
    extra_data_lists = {k: np.array(v, dtype=object)[sort_indices].tolist() for k, v in extra_data_lists.items()}

    if max_length is None:
        max_length = max(audio_data.size(-1) for audio_data in audio_data_lst)
    audio_tensor = []
    for audio_data in audio_data_lst:
        print(audio_data.shape)
        squeezed_data = audio_data.squeeze()
        if squeezed_data.dim() == 0:
            squeezed_data = squeezed_data.unsqueeze(0)
        if rand_append and random.random() < 0.5:
            x = (torch.zeros(max_length - audio_data.size(-1)), squeezed_data)
        else:
            x = (squeezed_data, torch.zeros(max_length - audio_data.size(-1)))
        audio_tensor.append(torch.cat(x, -1))
    return torch.stack(audio_tensor), extra_data_lists


def pad(data_list, element=0, max_length=None):
    if max_length is None:
        max_length = max(map(len, data_list))
    data_list = [x + [element] * (max_length - len(x)) for x in data_list]
    return data_list


class ZmuvTransform(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("total", torch.zeros(1))
        self.register_buffer("mean", torch.zeros(1))
        self.register_buffer("mean2", torch.zeros(1))

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
