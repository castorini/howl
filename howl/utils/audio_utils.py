import warnings

import librosa
import numpy as np
import torch


def silent_load(path: str, sample_rate: int = 16000, mono: bool = True) -> np.ndarray:
    """Load audio data using librosa without warning

    Args:
        path (str): path of the audio file
        sample_rate (int): sample rate which the audio will be loaded with
        mono (bool): if True, load the audio data as single-channel (mono) audio

    Returns:
        np.ndarray representing the audio data
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # TODO: by returning data in the first index,
        #       the returned data will be mono channel regardless of the original audio data
        return librosa.core.load(path, sr=sample_rate, mono=mono)[0]


def stride(
    audio_data: torch.Tensor, window_ms: int, stride_ms: int, sample_rate: int, drop_incomplete: bool = True
) -> torch.Tensor:
    """Stride over the audio data

    Args:
        audio_data (torch.Tensor): audio data to stride
        window_ms (int): size of the individual window in ms
        stride_ms (int): size of the stride (step) in ms
        sample_rate (int): sample rate of which the audio is loaded with
        drop_incomplete (bool): if True, the last window that is shorter than window_ms will be dropped

    Returns:
        generators returning single window at a time
    """
    chunk_sz = int(window_ms / 1000 * sample_rate)
    stride_sz = int(stride_ms / 1000 * sample_rate)
    curr_idx = 0
    while curr_idx < audio_data.size(-1):
        sliced = audio_data[..., curr_idx : curr_idx + chunk_sz]
        if sliced.size(-1) != chunk_sz and drop_incomplete:
            return
        yield sliced
        curr_idx += stride_sz
