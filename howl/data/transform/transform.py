import math
import random
from dataclasses import dataclass
from typing import Sequence

import librosa
import numpy as np
import torch
import torch.nn as nn
from torchaudio.transforms import ComputeDeltas, MelSpectrogram

from howl.data.common.example import EmplacableExample
from howl.data.common.sample import Sample
from howl.data.dataset.dataset import AudioClipDataset
from howl.data.transform.meyda import MeydaMelSpectrogram
from howl.settings import SETTINGS
from howl.utils import random_utils

__all__ = [
    "AugmentationParameter",
    "AugmentModule",
    "TimeshiftTransform",
    "TimestretchTransform",
    "NoiseTransform",
    "DatasetMixer",
    "StandardAudioTransform",
    "SpecAugmentTransform",
]


# pylint: disable=invalid-name
# pylint: disable=unused-argument

# TODO: this file needs to be separated into three
#  1) audio augmentation
#  2) spectrogram augmentation
#  3) standard audio to spectrogram transform


@dataclass
class AugmentationParameter:
    """AugmentationParameter"""

    domain: Sequence[float]
    name: str
    current_value_idx: int = None
    prob: float = 0.75
    enabled: bool = True

    def copy_from(self, op: "AugmentationParameter"):
        """copy_from"""
        self.current_value_idx = op.current_value_idx
        self.prob = op.prob
        self.enabled = op.enabled

    @property
    def magnitude(self):
        """magnitude"""
        return self.domain[self.current_value_idx]

    @classmethod
    def from_dict(cls, data_dict):
        """from_dict"""
        return cls(data_dict["domain"], data_dict["name"], data_dict["current_value_idx"], data_dict["prob"])


class AugmentModule(nn.Module):
    """Base torch module designed for augmentations"""

    def __init__(self, seed: int = None):
        """__init__"""
        super().__init__()
        self.augment_params = self.default_params
        self.rand = random if seed is None else random.Random(seed)
        self.seed = seed

    def reset_random(self):
        """Reset random number generator"""
        if self.seed is not None:
            random_utils.set_random_seed(self.seed)

    @property
    def default_params(self) -> Sequence[AugmentationParameter]:
        """Default parameters to use"""
        raise NotImplementedError

    def augment(self, param: AugmentationParameter, examples, **kwargs):
        """Applies the augmentation"""
        raise NotImplementedError

    def passthrough(self, examples, **kwargs):
        """Skips the augmentation"""
        return examples

    def forward(self, x, **kwargs):
        """Apply augmentation in training model, otherwise skips the augmentation"""
        for param in self.augment_params:
            if param.enabled and self.rand.random() < param.prob and self.training:
                if isinstance(x[0], Sample):
                    for sample in x:
                        self.augment_sample(param, sample, **kwargs)
                        # sample.audio_data = self.augment_audio_data(param, sample.audio_data, **kwargs)
                        # if sample.label is not None:
                        #     sample.label = self.augment_label(param, sample.label, **kwargs)
                else:  # Example augmentation, to be deprecated
                    x = self.augment(param, x, **kwargs)
            else:
                x = self.passthrough(x, **kwargs)
        return x


class TimeshiftTransform(AugmentModule):
    """Time-shift the audio data"""

    def __init__(self, sr=16000):
        super().__init__()
        self.sr = sr

    @property
    def default_params(self):
        """Shift magnitude in seconds"""
        return (AugmentationParameter([0.25, 0.5, 0.75, 1], "timeshift", 0),)

    @torch.no_grad()
    def augment(self, param: AugmentationParameter, examples: Sequence[EmplacableExample], **kwargs):
        """Apply time shift on the audio data"""
        new_examples = []
        for example in examples:
            w = min(int(self.rand.random() * param.magnitude * self.sr), int(0.5 * example.audio_data.size(-1)))
            audio_data = example.audio_data
            audio_data = (
                audio_data[..., w:] if self.rand.random() < 0.5 else audio_data[..., : example.audio_data.size(-1) - w]
            )
            new_examples.append(example.update_audio_data(audio_data))
        return new_examples

    def augment_sample(self, param: AugmentationParameter, sample: Sample, **kwargs):
        """Apply time shift (roll audio data)"""

        audio_data_size = sample.audio_data.size(-1)

        time_shift_mag = int(self.rand.random() * param.magnitude * self.sr)
        if sample.audio_data.size(-1) < 2 * time_shift_mag:
            time_shift_mag = int(0.5 * audio_data_size)

        # direction
        time_shift = time_shift_mag
        if self.rand.random() < 0.5:
            time_shift *= -1

        sample.audio_data = torch.roll(sample.audio_data, time_shift)

        # TODO: update labels if necessary
        # new_timestamp_label_map = {}
        # for timestamp, label in self.label_data.timestamp_label_map.items():
        #     new_timestamp = max(0, min(timestamp + time_shift, audio_data_size-1))
        #     new_timestamp_label_map[new_timestamp] = label
        #
        # sample.label.timestamp_label_map = new_timestamp_label_map


class TimestretchTransform(AugmentModule):
    """Time-stretch the audio data"""

    @property
    def default_params(self):
        """Shift magnitude; standard deviation"""
        return (AugmentationParameter([0.1, 0.2, 0.3], "timestretch", 1, prob=0.8),)

    @torch.no_grad()
    def augment(self, param, examples: Sequence[EmplacableExample], **kwargs):
        """Apply time stretch on the audio data"""
        new_examples = []
        for example in examples:
            # Stretch factor. If rate > 1, then the signal is sped up. If rate < 1, then the signal is slowed down.
            rate = np.clip(np.random.normal(1.0, param.magnitude), 0.3, 1.7)
            audio = torch.from_numpy(
                librosa.effects.time_stretch(example.audio_data.squeeze().cpu().numpy(), rate=rate)
            )
            new_examples.append(example.update_audio_data(audio, scale=1 / rate))
        return new_examples

    @torch.no_grad()
    def augment_sample(self, param: AugmentationParameter, sample: Sample, **kwargs):
        """Apply time stretch"""

        # Stretch factor. If rate > 1, then the signal is sped up. If rate < 1, then the signal is slowed down.
        stretch_rate = np.clip(np.random.normal(1.0, param.magnitude), 0.3, 1.7)

        sample.audio_data = torch.from_numpy(
            librosa.effects.time_stretch(sample.audio_data.squeeze().cpu().numpy(), rate=stretch_rate)
        )

        # # TODO: update labels if necessary
        # audio_data_size = sample.audio_data.size(-1)
        # scale = 1 / stretch_rate
        # new_timestamp_label_map = {}
        # for timestamp, label in self.label_data.timestamp_label_map.items():
        #     new_timestamp = min(scale*timestamp, audio_data_size-1)
        #     new_timestamp_label_map[new_timestamp] = label


class NoiseTransform(AugmentModule):
    """Add synthetic noise to the audio data"""

    @property
    def default_params(self):
        """Noise magnitude"""
        return (
            AugmentationParameter([0.0001, 0.00025, 0.0005, 0.001, 0.002], "white", 3),
            AugmentationParameter([1 / 20000, 1 / 15000, 1 / 10000, 1 / 5000, 1 / 2500], "salt_pepper", 2),
        )

    @torch.no_grad()
    def augment(self, param, examples: Sequence[EmplacableExample], **kwargs):
        """Add synthetic noise to the audio data"""
        new_examples = []
        for example in examples:
            waveform = example.audio_data
            if param.name == "white":
                strength = param.magnitude * self.rand.random()
                noise_mask = torch.empty_like(waveform).normal_(0, strength)
            else:
                prob = param.magnitude * self.rand.random()
                noise_mask = torch.empty_like(waveform).bernoulli_(prob / 2) - torch.empty_like(waveform).bernoulli_(
                    prob / 2
                )
            noise_mask.clamp_(-1, 1)
            waveform = (waveform + noise_mask).clamp_(-1, 1)
            new_examples.append(example.update_audio_data(waveform))
        return new_examples

    @torch.no_grad()
    def augment_sample(self, param: AugmentationParameter, sample: Sample, **kwargs):
        """Apply noise"""
        if param.name == "white":
            strength = param.magnitude * self.rand.random()
            noise_mask = torch.empty_like(sample.audio_data).normal_(0, strength)
        else:
            prob = param.magnitude * self.rand.random()
            noise_mask = torch.empty_like(sample.audio_data).bernoulli_(prob / 2) - torch.empty_like(
                sample.audio_data
            ).bernoulli_(prob / 2)
        noise_mask.clamp_(-1, 1)
        sample.audio_data = (sample.audio_data + noise_mask).clamp_(-1, 1)


class DatasetMixer(AugmentModule):
    """Augmentation by adding background noise"""

    def __init__(self, background_noise_dataset: AudioClipDataset, do_replace: bool = False, **kwargs):
        self.do_replace = do_replace
        super().__init__(**kwargs)
        self.dataset = background_noise_dataset

    @property
    def default_params(self):
        """Noise magnitude"""
        return (
            AugmentationParameter([0.1, 0.2, 0.3, 0.4, 0.5], "strength", 1),
            AugmentationParameter([0], "replace", 0, prob=0.1 if self.do_replace else 0),
        )

    @torch.no_grad()
    def augment(self, param, examples: Sequence[EmplacableExample], **kwargs):
        """Add background noise to the audio data"""
        new_examples = []
        for example in examples:
            waveform = example.audio_data
            bg_ex = self.rand.choice(self.dataset).audio_data.to(waveform.device)
            while bg_ex.size(-1) < waveform.size(-1):
                bg_ex = self.rand.choice(self.dataset).audio_data.to(waveform.device)
            b = self.rand.randint(waveform.size(-1), bg_ex.size(-1))
            a = b - waveform.size(-1)
            bg_audio = bg_ex[..., a:b]
            alpha = 1 if param.name == "replace" else self.rand.random() * param.magnitude
            mixed_wf = waveform * (1 - alpha) + bg_audio * alpha
            ex = example.update_audio_data(mixed_wf, new=alpha == 1)
            new_examples.append(ex)
        return new_examples

    @torch.no_grad()
    def augment_sample(self, param: AugmentationParameter, sample: Sample, **kwargs):
        """Add background noise"""
        bg_ex = self.rand.choice(self.dataset).audio_data.to(sample.audio_data.device)
        while bg_ex.size(-1) < sample.audio_data.size(-1):
            bg_ex = self.rand.choice(self.dataset).audio_data.to(sample.audio_data.device)
        b = self.rand.randint(sample.audio_data.size(-1), bg_ex.size(-1))
        a = b - sample.audio_data.size(-1)
        bg_audio = bg_ex[..., a:b]
        alpha = 1 if param.name == "replace" else self.rand.random() * param.magnitude
        sample.audio_data = sample.audio_data * (1 - alpha) + bg_audio * alpha


class StandardAudioTransform(AugmentModule):
    """Transformation to apply on the audio data"""

    def __init__(self):
        """__init__"""
        super().__init__()
        settings = SETTINGS.audio_transform
        if settings.use_meyda_spectrogram:
            self.spec_transform = MeydaMelSpectrogram(
                n_mels=settings.num_mels,
                sample_rate=settings.sample_rate,
                n_fft=settings.num_fft,
                hop_length=settings.hop_length,
            )
        else:
            self.spec_transform = MelSpectrogram(
                n_mels=settings.num_mels,
                sample_rate=settings.sample_rate,
                n_fft=settings.num_fft,
                hop_length=settings.hop_length,
            )

        self.vtlp_transform = apply_vtlp(
            MelSpectrogram(
                n_mels=settings.num_mels,
                sample_rate=settings.sample_rate,
                n_fft=settings.num_fft,
                hop_length=settings.hop_length,
            )
        )
        self.delta_transform = ComputeDeltas()

    @property
    def default_params(self) -> Sequence[AugmentationParameter]:
        """Parameter for vtlp augmentation"""
        return (AugmentationParameter([0], "vtlp", 0),)

    @torch.no_grad()
    def _execute_op(self, op, audio, mels_only=False, deltas_only=False):
        """Apply the transforms"""
        with torch.no_grad():
            log_mels = audio if deltas_only else op(audio).add_(1e-7).log_().contiguous()
            if mels_only:
                return log_mels
            deltas = self.delta_transform(log_mels)
            accels = self.delta_transform(deltas)
            return torch.stack((log_mels, deltas, accels), 1)

    def augment(self, param, examples: torch.Tensor, **kwargs):
        """Applies vtlp transform on mel spectrogram"""
        return self._execute_op(self.vtlp_transform, examples, **kwargs)

    def passthrough(self, examples: torch.Tensor, **kwargs):
        """Return mel spectrogram without applying additional transform"""
        return self._execute_op(self.spec_transform, examples, **kwargs)

    @torch.no_grad()
    def compute_lengths(self, length: torch.Tensor):
        """compute_lengths"""
        return (
            torch.div((length - self.spec_transform.win_length), self.spec_transform.hop_length, rounding_mode="floor")
            + 1
        ).long()


class SpecAugmentTransform(AugmentModule):
    """Augmentation designed for spectrogram"""

    @property
    def default_params(self) -> Sequence[AugmentationParameter]:
        """Default parameters for time masking and frequency masking"""
        return (
            AugmentationParameter([2, 5, 10, 20, 25], "sa_freq", 2),
            AugmentationParameter([10, 50, 75, 125, 150], "sa_time", 2),
        )

    def tmask(self, x, T):
        """Time Masking"""
        for idx in range(x.size(0)):
            t = self.rand.randrange(0, T)
            try:
                t0 = self.rand.randrange(0, x.size(3) - t)
            except ValueError:
                continue
            x[idx, :, :, t0 : t0 + t] = 0
        return x

    def fmask(self, x, F):
        """Frequency Masking"""
        for idx in range(x.size(0)):
            f = self.rand.randrange(0, F)
            f0 = self.rand.randrange(0, x.size(2) - f)
            x[idx, :, f0 : f0 + f] = 0
        return x

    @torch.no_grad()
    def augment(self, param, examples, **kwargs):
        """Apply SpecAugmentTransform"""
        with torch.no_grad():
            if param.name == "sa_freq":
                augmented_examples = self.fmask(examples, param.magnitude)
            elif param.name == "sa_time":
                augmented_examples = self.tmask(examples, param.magnitude)
            else:
                raise RuntimeError(f"Invalid parameter name for SpecAugmentTransform: {param.name}")
        return augmented_examples


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


def create_vtlp_fb_matrix(
    n_freqs: int,
    f_min: float,
    f_max: float,
    n_mels: int,
    sample_rate: int,
    alpha: float,
    f_hi: int = 4800,
    training: bool = True,
) -> torch.Tensor:
    """create_vtlp_fb_matrix"""
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
        f_pts[f_pts > f_hi * min(alpha, 1) / alpha] = S / 2 - (
            (S / 2 - f_hi * min(alpha, 1)) / (S / 2 - f_hi * min(alpha, 1) / alpha)
        ) * (S / 2 - f)
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
    """VtlpMelScale"""

    __constants__ = ["n_mels", "sample_rate", "f_min", "f_max"]

    def __init__(self, n_mels=128, sample_rate=16000, f_min=0.0, f_max=None, n_stft=None):
        """__init__"""
        super().__init__()
        self.n_mels = n_mels
        self.sample_rate = sample_rate
        self.f_max = f_max if f_max is not None else float(sample_rate // 2)
        self.f_min = f_min

        assert f_min <= self.f_max, "Require f_min: %f < f_max: %f" % (f_min, self.f_max)

    def forward(self, specgram):
        """forward"""
        # pack batch
        shape = specgram.size()
        specgram = specgram.reshape(-1, shape[-2], shape[-1])

        fb = create_vtlp_fb_matrix(
            specgram.size(1),
            self.f_min,
            self.f_max,
            self.n_mels,
            self.sample_rate,
            random.random() * 0.2 + 0.9,
            training=self.training,
        ).to(specgram.device)
        # (channel, frequency, time).transpose(...) dot (frequency, n_mels)
        # -> (channel, time, n_mels).transpose(...)
        mel_specgram = torch.matmul(specgram.transpose(1, 2), fb).transpose(1, 2)
        # unpack batch
        mel_specgram = mel_specgram.reshape(shape[:-2] + mel_specgram.shape[-2:])
        return mel_specgram


# END LICENSED BLOCK ###########


def apply_vtlp(mel_spectrogram: MelSpectrogram):
    """apply_vtlp"""
    s = mel_spectrogram
    s.mel_scale = VtlpMelScale(s.n_mels, s.sample_rate, s.f_min, s.f_max, s.n_fft // 2 + 1)
    return s
