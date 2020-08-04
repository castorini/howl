from librosa.filters import get_window
from librosa import filters
from librosa import util
import numpy as np
import torch


class MeydaMelSpectrogram:
    def __init__(
        self,
        n_fft: int = 512,
        n_mels: int = 80,
        sample_rate: int = 16000,
        hop_length: int = 200,
        f_max= 8000, # default
        f_min= 0, # default
        power = 2.0, # default
        win_length = None,
        window = 'hann', # default
        center = True,
        pad_mode = 'reflect', # default
        norm = None, # default for pytorch
        htk = True # default for pytorch
    ):
        self.n_fft = n_fft
        self.sample_rate = sample_rate
        self.pad_mode = pad_mode
        self.hop_length = hop_length
        self.power = power
        self.win_length = n_fft
        self.mel_basis = filters.mel(
            sr=sample_rate, 
            n_fft=n_fft,
            n_mels=n_mels, # mel filter
            fmin=f_min, # mel filter
            fmax=f_max,  # mel filter
            norm=norm,  # mel filter
            htk=htk
            )
        self.fft_window = get_window(window, self.win_length, fftbins=True).reshape((-1, 1))

    def fft(self, x):
        """A vectorized, non-recursive version of the Cooley-Tukey FFT"""
        x = np.asarray(x, dtype=float)
        N = x.shape[0]
        if np.log2(N) % 1 > 0:
            raise ValueError("size of x must be a power of 2")
        # N_min here is equivalent to the stopping condition above,
        # and should be a power of 2
        N_min = min(N, 32)
        # Perform an O[N^2] DFT on all length-N_min sub-problems at once
        n = np.arange(N_min)
        k = n[:, None]
        M = np.exp(-2j * np.pi * n * k / N_min)
        X = np.dot(M, x.reshape((N_min, -1)))
        # build-up each level of the recursive calculation all at once
        while X.shape[0] < N:
            X_even = X[:, :int(X.shape[1] / 2)]
            X_odd = X[:, int(X.shape[1] / 2):]
            factor = np.exp(-1j * np.pi * np.arange(X.shape[0])
                            / X.shape[0])[:, None]
            X = np.vstack([X_even + factor * X_odd,
                           X_even - factor * X_odd])
        return X.ravel()

    def spectrogram(self, audio_data):
        # Pad the time series so that frames are centered
        y = np.pad(audio_data, int(self.n_fft // 2), mode=self.pad_mode)
        y_frames = util.frame(y, frame_length=self.n_fft, hop_length=self.hop_length)
        windowed = (self.fft_window * y_frames).T
        fft_matrix = []
        for frame in windowed:
            complex_fft = self.fft(frame)
            amp_spectr = np.sqrt(((complex_fft.real ** 2) + (complex_fft.imag ** 2)))[:self.n_fft // 2 + 1]
            fft_matrix.append(amp_spectr)
        fft_matrix = np.stack(fft_matrix)
        return np.abs(fft_matrix)**self.power

    def __call__(self, audio_data):
        device = audio_data.device
        audio_data = audio_data.cpu()
        batch = []
        for sample in audio_data:
            s = self.spectrogram(sample.squeeze())
            batch.append(torch.tensor(np.dot(self.mel_basis, s.T)))
        batch = torch.stack(batch).float().to(device)
        return batch