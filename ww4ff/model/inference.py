from typing import List
import itertools
import logging
import time

from pydantic import BaseSettings
import numpy as np
import torch
import torch.nn as nn

from ww4ff.data.transform import ZmuvTransform, StandardAudioTransform


__all__ = ['InferenceEngine', 'InferenceEngineSettings']


class InferenceEngineSettings(BaseSettings):
    inference_threshold: float = 0.5
    inference_alpha: float = 0.9
    inference_weights: List[float] = None
    inference_sequence: List[int] = None
    inference_window: float = 2.5


class InferenceEngine:
    def __init__(self,
                 model: nn.Module,
                 zmuv_transform: ZmuvTransform,
                 negative_label: int,
                 settings: InferenceEngineSettings = InferenceEngineSettings(),
                 time_provider=time.time):
        self.model = model
        self.zmuv = zmuv_transform
        self.std = StandardAudioTransform().eval()
        self.alpha = settings.inference_alpha
        self.threshold = settings.inference_threshold
        inference_weights = 1 if settings.inference_weights is None else np.array(settings.inference_weights)
        self.inference_weights = inference_weights
        self.negative_label = negative_label
        self.window = settings.inference_window
        self.sequence = settings.inference_sequence
        self.sequence_str = ''.join(map(str, settings.inference_sequence))
        self.history = []
        self._decoded_history = ''

    def reset(self):
        pass

    @property
    def sequence_present(self) -> bool:
        if not self.sequence_str:
            return False
        self._prune_history()
        try:
            self._decoded_history.index(self.sequence_str)
            return True
        except ValueError:
            return False

    def _prune_history(self, curr_time: float):
        self.history = list(itertools.dropwhile(lambda x: curr_time - x[0] > self.window, self.history))
        self._decoded_history = ''.join(map(str, list(x[1] for x in self.history)))

    @torch.no_grad()
    def infer(self,
              x: torch.Tensor,
              lengths: torch.Tensor = None,
              curr_time: float = None) -> int:
        if curr_time is None:
            curr_time = time.time()
        self.std = self.std.to(x.device)
        if lengths is None:
            lengths = torch.tensor([x.size(-1)]).to(x.device)
        lengths = self.std.compute_lengths(lengths)
        x = self.zmuv(self.std(x.unsqueeze(0)))
        p = self.model(x, lengths).softmax(-1)[0].cpu().numpy()
        p *= self.inference_weights
        p = p / p.sum()
        logging.debug([f'{x:.3f}' for x in p.tolist()])
        prob = 1 - p[self.negative_label]
        if prob > self.threshold:
            p[self.negative_label] = 0
            label, label_val = np.argmax(p), np.max(p)
            if self.sequence_str:
                self.history.append((curr_time, label))
        else:
            label = self.negative_label
        if self.sequence_str:
            self._prune_history(curr_time)
        return label