from typing import Sequence, Iterable
import random

import librosa.effects as effects
import torch
import torch.nn as nn

from howl.data.dataset import WakeWordClipExample, ClassificationBatch, EmplacableExample


__all__ = ['Composition',
           'compose',
           'ZmuvTransform',
           'random_slice',
           'WakeWordBatchifier',
           'batchify',
           'identity',
           'trim',
           'truncate_length']


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


def truncate_length(examples: Sequence[EmplacableExample], length: int = None):
    return [ex.emplaced_audio_data(ex.audio_data[..., :length]) for ex in examples]


def batchify(examples: Sequence[EmplacableExample], label_provider=None):
    examples = sorted(examples, key=lambda x: x.audio_data.size()[-1], reverse=True)
    lengths = torch.tensor([ex.audio_data.size(-1) for ex in examples])
    max_length = max(ex.audio_data.size(-1) for ex in examples)
    audio_tensor = [torch.cat((ex.audio_data.squeeze(), torch.zeros(max_length - ex.audio_data.size(-1))), -1) for
                    ex in examples]
    audio_tensor = torch.stack(audio_tensor)
    labels = torch.tensor(list(map(label_provider, examples))) if label_provider else None
    return ClassificationBatch(audio_tensor, labels, lengths)


class WakeWordBatchifier:
    def __init__(self,
                 negative_label: int,
                 positive_sample_prob: float = 1,
                 window_size_ms: int = 500,
                 sample_rate: int = 16000,
                 positive_delta_ms: int = 150,
                 eps_ms: int = 20,
                 pad_to_window: bool = True):
        self.positive_sample_prob = positive_sample_prob
        self.window_size_ms = window_size_ms
        self.sample_rate = sample_rate
        self.negative_label = negative_label
        self.positive_delta_ms = positive_delta_ms
        self.eps_ms = eps_ms
        self.pad_to_window = pad_to_window

    def __call__(self, examples: Sequence[WakeWordClipExample]) -> ClassificationBatch:
        new_examples = []
        for ex in examples:
            if not ex.label_data.timestamp_label_map:
                new_examples.append((self.negative_label,
                                     random_slice([ex], int(self.sample_rate * self.window_size_ms / 1000))[0]))
                continue
            select_negative = random.random() > self.positive_sample_prob
            if not select_negative:
                end_ms, label = random.choice(list(ex.label_data.timestamp_label_map.items()))
                end_ms_rand = end_ms + (random.random() * self.eps_ms)
                b = int((end_ms_rand / 1000) * self.sample_rate)
                a = max(b - int((self.window_size_ms / 1000) * self.sample_rate), 0)
                if random.random() < 0:
                    closest_ms = min(filter(lambda k: end_ms - k > 0, ex.label_data.timestamp_label_map.keys()),
                                     key=lambda k: end_ms - k,
                                     default=-1)
                    if closest_ms >= 0:
                        a = max(a, int((closest_ms / 1000) * self.sample_rate))
                if b - a < 0:
                    select_negative = True
                else:
                    new_examples.append((label, ex.emplaced_audio_data(ex.audio_data[..., a:b])))
            if select_negative:
                positive_intervals = [(v - self.positive_delta_ms, v + self.positive_delta_ms)
                                      for v in ex.label_data.timestamp_label_map.values()]
                positive_intervals = sorted(positive_intervals, key=lambda x: x[0])
                negative_intervals = []
                last_positive = 0
                for a, b in positive_intervals:
                    if last_positive < a:
                        negative_intervals.append((last_positive, a))
                    last_positive = b
                negative_intervals.append((b, int(len(ex.audio_data) / 16000 * 1000)))
                a, b = random.choice(negative_intervals)
                window_sz_samples = int(self.window_size_ms / 1000 * self.sample_rate)
                a = int(a / 1000 * self.sample_rate)
                b = int(b / 1000 * self.sample_rate)
                if b - a > window_sz_samples:
                    a = random.randint(0, b - window_sz_samples)
                    b = a + window_sz_samples
                new_examples.append((self.negative_label, ex.emplaced_audio_data(ex.audio_data[..., a:b])))
        new_examples = sorted(new_examples, key=lambda x: x[1].audio_data.size()[-1], reverse=True)
        lengths = torch.tensor([ex.audio_data.size(-1) for _, ex in new_examples])
        if self.pad_to_window:
            max_length = int(self.sample_rate * self.window_size_ms / 1000)
        else:
            max_length = max(ex.audio_data.size(-1) for _, ex in new_examples)
        audio_tensor = []
        for _, ex in new_examples:
            if random.random() < 0.5:
                x = (torch.zeros(max_length - ex.audio_data.size(-1)), ex.audio_data.squeeze())
            else:
                x = (ex.audio_data.squeeze(), torch.zeros(max_length - ex.audio_data.size(-1)))
            audio_tensor.append(torch.cat(x, -1))

        audio_tensor = torch.stack(audio_tensor)
        labels_tensor = torch.tensor([lidx for lidx, _ in new_examples])
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
