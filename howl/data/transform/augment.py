from dataclasses import dataclass
from typing import Sequence
import math
import random

from torchaudio.transforms import MelSpectrogram, ComputeDeltas
import librosa
import torch
import torch.nn as nn

from howl.data.dataset import EmplacableExample, WakeWordClipExample, AudioClipDataset
from howl.settings import SETTINGS
from .meyda import MeydaMelSpectrogram


__all__ = ['AugmentationParameter',
           'AugmentModule',
           'TimeshiftTransform',
           'TimestretchTransform',
           'NoiseTransform',
           'DatasetMixer',
           'StandardAudioTransform',
           'SpecAugmentTransform',
           'NegativeSampleTransform']


@dataclass
class AugmentationParameter:
    domain: Sequence[float]
    name: str
    current_value_idx: int = None
    prob: float = 0.75
    enabled: bool = True

    def copy_from(self, op: 'AugmentationParameter'):
        self.current_value_idx = op.current_value_idx
        self.prob = op.prob
        self.enabled = op.enabled

    @property
    def magnitude(self):
        return self.domain[self.current_value_idx]

    @classmethod
    def from_dict(cls, data_dict):
        return cls(data_dict['domain'], data_dict['name'], data_dict['current_value_idx'], data_dict['prob'])


class AugmentModule(nn.Module):
    def __init__(self, seed: int = None):
        super().__init__()
        self.augment_params = self.default_params
        self.rand = random if seed is None else random.Random(seed)
        self.seed = seed

    def reset_random(self):
        if self.seed is not None:
            self.rand = random.Random(self.seed)

    @property
    def default_params(self) -> Sequence[AugmentationParameter]:
        raise NotImplementedError

    def augment(self, param: AugmentationParameter, x, **kwargs):
        raise NotImplementedError

    def passthrough(self, x, **kwargs):
        return x

    def forward(self, x, **kwargs):
        for param in self.augment_params:
            if param.enabled and self.rand.random() < param.prob and self.training:
                x = self.augment(param, x, **kwargs)
            else:
                x = self.passthrough(x, **kwargs)
        return x


class NegativeSampleTransform(AugmentModule):
    @property
    def default_params(self):
        return AugmentationParameter([0.2, 0.3, 0.4, 0.5], 'chunk_size', 1, prob=0.3),

    @torch.no_grad()
    def augment(self, param: AugmentationParameter, examples: Sequence[WakeWordClipExample], **kwargs):
        new_examples = []
        for example in examples:
            audio_data = example.audio_data[..., :int(example.audio_data.size(-1) * param.magnitude)]
            example = example.emplaced_audio_data(audio_data)
            example.contains_wake_word = False
            new_examples.append(example)
        return new_examples


class TimeshiftTransform(AugmentModule):
    def __init__(self, sr=16000):
        super().__init__()
        self.sr = sr

    @property
    def default_params(self):
        return AugmentationParameter([0.25, 0.5, 0.75, 1], 'timeshift', 0),

    @torch.no_grad()
    def augment(self, param: AugmentationParameter, examples: Sequence[EmplacableExample], **kwargs):
        new_examples = []
        for example in examples:
            w = min(int(self.rand.random() * param.magnitude * self.sr), int(0.5 * example.audio_data.size(-1)))
            audio_data = example.audio_data
            audio_data = audio_data[..., w:] if self.rand.random() < 0.5 else audio_data[..., :example.audio_data.size(-1) - w]
            new_examples.append(example.emplaced_audio_data(audio_data))
        return new_examples


class TimestretchTransform(AugmentModule):
    @property
    def default_params(self):
        return AugmentationParameter([0.025, 0.05, 0.15, 0.2, 0.25], 'timestretch', 2, prob=0.3),

    @torch.no_grad()
    def augment(self, param, examples: Sequence[EmplacableExample], **kwargs):
        new_examples = []
        for example in examples:
            rate = 1.5  # np.clip(np.random.normal(1.1, param.magnitude), 0.8, 2)
            audio = torch.from_numpy(librosa.effects.time_stretch(example.audio_data.squeeze().cpu().numpy(), rate))
            new_examples.append(example.emplaced_audio_data(audio, scale=1 / rate))
        return new_examples


class NoiseTransform(AugmentModule):
    @property
    def default_params(self):
        return AugmentationParameter([0.0001, 0.00025, 0.0005, 0.001, 0.002], 'white', 3),\
               AugmentationParameter([1 / 20000, 1 / 15000, 1 / 10000, 1 / 5000, 1 / 2500], 'salt_pepper', 2)

    @torch.no_grad()
    def augment(self, param, examples: Sequence[EmplacableExample], **kwargs):
        new_examples = []
        for example in examples:
            waveform = example.audio_data
            if param.name == 'white':
                strength = param.magnitude * self.rand.random()
                noise_mask = torch.empty_like(waveform).normal_(0, strength)
            else:
                prob = param.magnitude * self.rand.random()
                noise_mask = torch.empty_like(waveform).bernoulli_(prob / 2) - torch.empty_like(waveform).bernoulli_(prob / 2)
            noise_mask.clamp_(-1, 1)
            waveform = (waveform + noise_mask).clamp_(-1, 1)
            new_examples.append(example.emplaced_audio_data(waveform))
        return new_examples


class DatasetMixer(AugmentModule):
    def __init__(self,
                 background_noise_dataset: AudioClipDataset,
                 do_replace: bool = False,
                 **kwargs):
        self.do_replace = do_replace
        super().__init__(**kwargs)
        self.dataset = background_noise_dataset

    @property
    def default_params(self):
        return (AugmentationParameter([0.1, 0.2, 0.3, 0.4, 0.5], 'strength', 1),
                AugmentationParameter([0], 'replace', 0, prob=0.1 if self.do_replace else 0))

    @torch.no_grad()
    def augment(self, param, examples: Sequence[EmplacableExample], **kwargs):
        new_examples = []
        for example in examples:
            waveform = example.audio_data
            bg_ex = self.rand.choice(self.dataset).audio_data.to(waveform.device)
            while bg_ex.size(-1) < waveform.size(-1):
                bg_ex = self.rand.choice(self.dataset).audio_data.to(waveform.device)
            b = self.rand.randint(waveform.size(-1), bg_ex.size(-1))
            a = b - waveform.size(-1)
            bg_audio = bg_ex[..., a:b]
            alpha = 1 if param.name == 'replace' else self.rand.random() * param.magnitude
            mixed_wf = waveform * (1 - alpha) + bg_audio * alpha
            ex = example.emplaced_audio_data(mixed_wf, new=alpha == 1)
            new_examples.append(ex)
        return new_examples


class StandardAudioTransform(AugmentModule):
    def __init__(self):
        super().__init__()
        settings = SETTINGS.audio_transform
        if settings.use_meyda_spectrogram:
            self.spec_transform = MeydaMelSpectrogram(n_mels=settings.num_mels,
                                                      sample_rate=settings.sample_rate,
                                                      n_fft=settings.num_fft,
                                                      hop_length=settings.hop_length)
        else:
            self.spec_transform = MelSpectrogram(n_mels=settings.num_mels,
                                                 sample_rate=settings.sample_rate,
                                                 n_fft=settings.num_fft,
                                                 hop_length=settings.hop_length)

        self.vtlp_transform = apply_vtlp(MelSpectrogram(n_mels=settings.num_mels,
                                                        sample_rate=settings.sample_rate,
                                                        n_fft=settings.num_fft,
                                                        hop_length=settings.hop_length))
        self.delta_transform = ComputeDeltas()

    @property
    def default_params(self) -> Sequence[AugmentationParameter]:
        return AugmentationParameter([0], 'vtlp', 0),

    @torch.no_grad()
    def _execute_op(self, op, audio, mels_only=False, deltas_only=False):
        with torch.no_grad():
            log_mels = audio if deltas_only else op(audio).add_(1e-7).log_().contiguous()
            if mels_only:
                return log_mels
            deltas = self.delta_transform(log_mels)
            accels = self.delta_transform(deltas)
            return torch.stack((log_mels, deltas, accels), 1)

    def augment(self, param, audio: torch.Tensor, **kwargs):
        return self._execute_op(self.vtlp_transform, audio, **kwargs)

    def passthrough(self, audio: torch.Tensor, **kwargs):
        return self._execute_op(self.spec_transform, audio, **kwargs)

    @torch.no_grad()
    def compute_lengths(self, length: torch.Tensor):
        return ((length - self.spec_transform.win_length) // self.spec_transform.hop_length + 1).long()


class SpecAugmentTransform(AugmentModule):
    @property
    def default_params(self) -> Sequence[AugmentationParameter]:
        return AugmentationParameter([2, 5, 10, 20, 25], 'sa_freq', 2),\
               AugmentationParameter([10, 50, 75, 125, 150], 'sa_time', 2)

    def tmask(self, x, T):
        for idx in range(x.size(0)):
            t = self.rand.randrange(0, T)
            try:
                t0 = self.rand.randrange(0, x.size(3) - t)
            except ValueError:
                continue
            x[idx, :, :, t0:t0 + t] = 0
        return x

    def fmask(self, x, F):
        for idx in range(x.size(0)):
            f = self.rand.randrange(0, F)
            f0 = self.rand.randrange(0, x.size(2) - f)
            x[idx, :, f0:f0 + f] = 0
        return x

    @torch.no_grad()
    def augment(self, param, x, **kwargs):
        with torch.no_grad():
            if param.name == 'sa_freq':
                return self.fmask(x, param.magnitude)
            elif param.name == 'sa_time':
                return self.tmask(x, param.magnitude)
        return x


# BEGIN LICENSED BLOCK ##########

"""
BSD 2-Clause License

Copyright (c) 2017 Facebook Inc. (Soumith Chintala), 
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

def create_vtlp_fb_matrix(n_freqs, f_min, f_max, n_mels, sample_rate, alpha, f_hi=4800, training=True):
    # type: (int, float, float, int, int, float, int, bool) -> torch.Tensor
    # freq bins
    # Equivalent filterbank construction by Librosa
    S = sample_rate
    all_freqs = torch.linspace(0, sample_rate // 2, n_freqs)

    # calculate mel freq bins
    # hertz to mel(f) is 2595. * math.log10(1. + (f / 700.))
    m_min = 2595.0 * math.log10(1.0 + (f_min / 700.0))
    m_max = 2595.0 * math.log10(1.0 + (f_max / 700.0))
    m_pts = torch.linspace(m_min, m_max, n_mels + 2)
    # mel to hertz(mel) is 700. * (10**(mel / 2595.) - 1.)
    f_pts = 700.0 * (10 ** (m_pts / 2595.0) - 1.0)
    if training:
        f_pts[f_pts <= f_hi * min(alpha, 1) / alpha] *= alpha
        f = f_pts[f_pts > f_hi * min(alpha, 1) / alpha]
        f_pts[f_pts > f_hi * min(alpha, 1) / alpha] = S / 2 - ((S / 2 - f_hi * min(alpha, 1)) /
                                                               (S / 2 - f_hi * min(alpha, 1) / alpha)) * (S / 2 - f)
    # calculate the difference between each mel point and each stft freq point in hertz
    f_diff = f_pts[1:] - f_pts[:-1]  # (n_mels + 1)
    slopes = f_pts.unsqueeze(0) - all_freqs.unsqueeze(1)  # (n_freqs, n_mels + 2)
    # create overlapping triangles
    zero = torch.zeros(1)
    down_slopes = (-1.0 * slopes[:, :-2]) / f_diff[:-1]  # (n_freqs, n_mels)
    up_slopes = slopes[:, 2:] / f_diff[1:]  # (n_freqs, n_mels)
    fb = torch.max(zero, torch.min(down_slopes, up_slopes))
    return fb


class VtlpMelScale(nn.Module):

    __constants__ = ['n_mels', 'sample_rate', 'f_min', 'f_max']

    def __init__(self, n_mels=128, sample_rate=16000, f_min=0., f_max=None, n_stft=None):
        super().__init__()
        self.n_mels = n_mels
        self.sample_rate = sample_rate
        self.f_max = f_max if f_max is not None else float(sample_rate // 2)
        self.f_min = f_min

        assert f_min <= self.f_max, 'Require f_min: %f < f_max: %f' % (f_min, self.f_max)

    def forward(self, specgram):
        # pack batch
        shape = specgram.size()
        specgram = specgram.reshape(-1, shape[-2], shape[-1])

        fb = create_vtlp_fb_matrix(specgram.size(1), self.f_min, self.f_max, self.n_mels, self.sample_rate,
                                   random.random() * 0.2 + 0.9, training=self.training).to(specgram.device)
        # (channel, frequency, time).transpose(...) dot (frequency, n_mels)
        # -> (channel, time, n_mels).transpose(...)
        mel_specgram = torch.matmul(specgram.transpose(1, 2), fb).transpose(1, 2)
        # unpack batch
        mel_specgram = mel_specgram.reshape(shape[:-2] + mel_specgram.shape[-2:])
        return mel_specgram

# END LICENSED BLOCK ###########


def apply_vtlp(mel_spectrogram: MelSpectrogram):
    s = mel_spectrogram
    s.mel_scale = VtlpMelScale(s.n_mels, s.sample_rate, s.f_min, s.f_max, s.n_fft // 2 + 1)
    return s