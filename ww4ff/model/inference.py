from collections import deque

import numpy as np
import torch
import torch.nn as nn

from ww4ff.data.transform import ZmuvTransform, StandardAudioTransform


__all__ = ['InferenceEngine']


class InferenceEngine:
    def __init__(self,
                 model: nn.Module,
                 zmuv_transform: ZmuvTransform,
                 alpha: float = 0.9,
                 threshold: float = 0.8):
        self.model = model
        self.zmuv = zmuv_transform
        self.std = StandardAudioTransform()
        self.alpha = alpha
        self.threshold = threshold
        self.value = 0

    def reset(self):
        self.value = 0

    @torch.no_grad()
    def infer(self, x: torch.Tensor, lengths: torch.Tensor = None) -> bool:
        self.std = self.std.to(x.device)
        if lengths is None:
            lengths = torch.tensor([x.size(-1)]).to(x.device)
        lengths = self.std.compute_lengths(lengths)
        x = self.zmuv(self.std(x.unsqueeze(0)))
        p = self.model(x, lengths).softmax(-1)[0, 1].item()
        self.value = self.value * (1 - self.alpha) + self.alpha * p
        return self.value > self.threshold
