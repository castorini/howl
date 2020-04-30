import warnings

import librosa


def silent_load(path: str, sr: int = 16000, mono: bool = True):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return librosa.core.load(path, sr=sr, mono=mono)[0]
