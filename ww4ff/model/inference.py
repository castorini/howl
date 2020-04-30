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
                 history_length: int = 10,
                 threshold: float = 0.7):
        self.model = model
        self.zmuv = zmuv_transform
        self.std = StandardAudioTransform()
        self.prob_outputs = deque()
        self.history_length = history_length
        self.threshold = threshold

    def reset(self):
        self.prob_outputs = deque()

    @torch.no_grad()
    def infer(self, x: torch.Tensor, lengths: torch.Tensor = None) -> bool:
        self.std = self.std.to(x.device)
        if lengths is None:
            lengths = torch.tensor([x.size(-1)]).to(x.device)
        lengths = self.std.compute_lengths(lengths)
        x = self.zmuv(self.std(x.unsqueeze(0)))
        p = self.model(x, lengths).softmax(-1)[0, 1].item()
        self.prob_outputs.append(p)
        if len(self.prob_outputs) > self.history_length:
            self.prob_outputs.popleft()

        mean_val = np.mean(list(self.prob_outputs))
        return mean_val > self.threshold
