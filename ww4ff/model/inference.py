from typing import List
import logging

from pydantic import BaseSettings
import numpy as np
import torch
import torch.nn as nn

from ww4ff.data.transform import ZmuvTransform, StandardAudioTransform


__all__ = ['InferenceEngine', 'InferenceEngineSettings']


class InferenceEngineSettings(BaseSettings):
    inference_threshold: float = 0.9
    inference_alpha: float = 0.9
    inference_weights: List[float] = None


class InferenceEngine:
    def __init__(self,
                 model: nn.Module,
                 zmuv_transform: ZmuvTransform,
                 negative_label: int,
                 settings: InferenceEngineSettings = InferenceEngineSettings()):
        self.model = model
        self.zmuv = zmuv_transform
        self.std = StandardAudioTransform().eval()
        self.alpha = settings.inference_alpha
        self.threshold = settings.inference_threshold
        self.value = 0
        inference_weights = 1 if settings.inference_weights is None else np.array(settings.inference_weights)
        self.inference_weights = inference_weights
        self.negative_label = negative_label

    def reset(self):
        self.value = 0

    @torch.no_grad()
    def infer(self, x: torch.Tensor, lengths: torch.Tensor = None) -> int:
        self.std = self.std.to(x.device)
        if lengths is None:
            lengths = torch.tensor([x.size(-1)]).to(x.device)
        lengths = self.std.compute_lengths(lengths)
        x = self.zmuv(self.std(x.unsqueeze(0)))
        p = self.model(x, lengths).softmax(-1)[0].cpu().numpy()
        p *= self.inference_weights
        p = p / p.sum()
        logging.debug([f'{x:.3f}' for x in p[:4].tolist()])
        prob = 1 - p[self.negative_label]
        self.value = self.value * (1 - self.alpha) + self.alpha * prob
        if self.value > self.threshold:
            return np.argmax(p)
        return self.negative_label
