import warnings

import librosa
import torch


def silent_load(path: str, sr: int = 16000, mono: bool = True):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return librosa.core.load(path, sr=sr, mono=mono)[0]


def stride(audio: torch.Tensor, chunk_ms: int, stride_ms: int, sample_rate: int) -> torch.Tensor:
    chunk_sz = int(chunk_ms / 1000 * sample_rate)
    stride_sz = int(stride_ms / 1000 * sample_rate)
    curr_idx = 0
    while curr_idx < audio.size(-1):
        yield audio[..., curr_idx:curr_idx + chunk_sz]
        curr_idx += stride_sz
