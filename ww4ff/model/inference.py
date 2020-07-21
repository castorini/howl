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
    inference_weights: List[float] = None
    inference_sequence: List[int] = None
    inference_window_ms: float = 2000  # look at last 2 seconds
    smoothing_window_ms: float = 50  # prediction smoothed over 1 seconds
    tolerance_window_ms: float = 500  # negative label between words are acceptable for 0.3 seconds


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
        inference_weights = 1 if settings.inference_weights is None else np.array(settings.inference_weights)
        self.inference_weights = inference_weights
        self.negative_label = negative_label
        self.inference_window_ms = settings.inference_window_ms
        self.smoothing_window_ms = settings.smoothing_window_ms
        self.tolerance_window_ms = settings.tolerance_window_ms
        self.sequence = settings.inference_sequence
        self.time_provider = time_provider
        self.pred_history = []
        self.label_history = []

    def append_label(self, label: int, curr_time: float = None):
        if not curr_time:
            curr_time = self.time_provider() * 1000
        self.label_history.append((curr_time, label))

    def sequence_present(self, curr_time: float = None) -> bool:
        if not self.sequence:
            return False
        if len(self.sequence) == 0:
            return True

        if not curr_time:
            curr_time = self.time_provider() * 1000

        self.label_history = list(itertools.dropwhile(lambda x: curr_time - x[0] > self.inference_window_ms,
                                                      self.label_history))  # drop entries that are old

        # finite state machine for detecting the sequence
        curr_label = None
        target_state = 0
        last_valid_timestamp = 0

        for history in self.label_history:
            curr_timestamp, label = history
            target_label = self.sequence[target_state]
            if label == target_label:
                # move to next state
                target_state += 1
                if target_state == len(self.sequence):
                    # goal state is reached
                    return True
                curr_label = self.sequence[target_state - 1]
                last_valid_timestamp = curr_timestamp
            elif label == curr_label:
                # label has not changed, only update last_valid_timestamp
                last_valid_timestamp = curr_timestamp
            elif last_valid_timestamp + self.tolerance_window_ms < curr_timestamp:
                # out of tolerance window, start from the first state
                curr_label = None
                target_state = 0
                last_valid_timestamp = 0
        return False

    def _get_prediction(self,
                        curr_time: float) -> int:
        # drop out-dated entries
        self.pred_history = list(itertools.dropwhile(lambda x: curr_time - x[0] > self.smoothing_window_ms, self.pred_history))
        lattice = np.vstack([t for _, t in self.pred_history])
        max_label = np.max(lattice, 0).argmax()
        self.label_history.append((curr_time, max_label))
        return max_label

    @torch.no_grad()
    def infer(self,
              x: torch.Tensor,
              lengths: torch.Tensor = None,
              curr_time: float = None) -> int:

        if not curr_time:
            curr_time = self.time_provider() * 1000

        self.std = self.std.to(x.device)
        if lengths is None:
            lengths = torch.tensor([x.size(-1)]).to(x.device)
        lengths = self.std.compute_lengths(lengths)
        x = self.zmuv(self.std(x.unsqueeze(0)))
        p = self.model(x, lengths).softmax(-1)[0].cpu().numpy()

        p *= self.inference_weights
        p = p / p.sum()
        logging.debug([f'{x:.3f}' for x in p.tolist()])

        self.pred_history.append((curr_time, p))
        label = self._get_prediction(curr_time)
        return label
