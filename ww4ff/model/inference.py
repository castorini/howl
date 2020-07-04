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
    inference_threshold: float = 0.2
    inference_alpha: float = 0.9
    inference_weights: List[float] = None
    inference_sequence: List[int] = None
    inference_window: float = 1


class InferenceEngine:
    def __init__(self,
                 model: nn.Module,
                 zmuv_transform: ZmuvTransform,
                 num_labels: int,
                 negative_label: int,
                 settings: InferenceEngineSettings = InferenceEngineSettings(),
                 time_provider=time.time):
        self.model = model
        self.zmuv = zmuv_transform
        self.std = StandardAudioTransform().eval()
        self.alpha = settings.inference_alpha
        self.threshold = settings.inference_threshold
        self.value = 0
        inference_weights = 1 if settings.inference_weights is None else np.array(settings.inference_weights)
        self.inference_weights = inference_weights
        self.num_labels = num_labels
        self.negative_label = negative_label
        self.window = settings.inference_window
        self.sequence = settings.inference_sequence
        self.sequence_str = ''.join(map(str, settings.inference_sequence))
        self.label_history = []
        self._decoded_history = ''
        self.time_provider = time_provider

        self.pred_history = [];
        self.smoothed_history = [];

    def reset(self):
        self.value = 0

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


    def _update_label_history(self):
        self.label_history = list(itertools.dropwhile(lambda x: self.time_provider() - x[0] > self.window, self.label_history)) # drop entries that are old

        label_sequence = [str(self.negative_label)]
        for label in self.label_history:
            if label_sequence[-1] != str(label):
                label_sequence.append(str(label))

        self._decoded_history = ''.join(label_sequence)


    def _update_pred_history(self):
        # drop out-dated entries
        curr_time = self.time_provider()
        self.pred_history = list(itertools.dropwhile(lambda x: curr_time - x[0] > self.window, self.pred_history))

        accum_history = torch.zeros(self.pred_history[0][1].shape)
        for x in self.pred_history:
            accum_history += x[1]

        self.smoothed_history.append((curr_time, accum_history))


    def _get_prediction(self) -> int:
        # drop out-dated entries
        curr_time = self.time_provider()
        self.smoothed_history = list(itertools.dropwhile(lambda x: curr_time - x[0] > self.window, self.smoothed_history))

        accum_smoothed_history = torch.zeros(self.smoothed_history[0][1].shape)
        for x in self.smoothed_history:
            accum_smoothed_history += x[1]

        final_pred = accum_smoothed_history.argmax();

        self.label_history.append(curr_time, final_pred)
        self._update_label_history()

        return final_pred;


    @torch.no_grad()
    def infer(self,
              x: torch.Tensor,
              lengths: torch.Tensor = None) -> int:
        self.std = self.std.to(x.device)
        if lengths is None:
            lengths = torch.tensor([x.size(-1)]).to(x.device)
        lengths = self.std.compute_lengths(lengths)
        x = self.zmuv(self.std(x.unsqueeze(0)))
        p = self.model(x, lengths).softmax(-1)[0].cpu().numpy()

        p *= self.inference_weights

        # normalize
        p = p / p.sum()
        logging.debug([f'{x:.3f}' for x in p.tolist()])

        self.pred_history.append((self.time_provider(), p))

        self._update_pred_history()
        label = self._get_prediction()
        
        # TODO:: current time must be passed in from outside

        # prob = 1 - p[self.negative_label] # sum of all the positive probability
        # self.value = self.value * (1 - self.alpha) + self.alpha * prob # windowing

        # if self.value > self.threshold: # when it's greater than the threshold
        #     label = np.argmax(p) # largest label
        #     if self.sequence_str: # largest label
        #         self.label_history.append((self.time_provider(), label))
        # else: # other wise negative sample
        #     label = self.negative_label
        # if self.sequence_str:
        #     self._prune_history()
        return label
