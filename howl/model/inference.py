import itertools
import logging
import time

import numpy as np
import torch
import torch.nn.functional as F

from howl.context import InferenceContext
from howl.data.transform import StandardAudioTransform, ZmuvTransform
from howl.model import RegisteredModel
from howl.settings import SETTINGS
from howl.utils.audio import stride

__all__ = ["FrameInferenceEngine", "InferenceEngine", "SequenceInferenceEngine"]


class InferenceEngine:
    def __init__(
        self, model: RegisteredModel, zmuv_transform: ZmuvTransform, context: InferenceContext, time_provider=time.time
    ):

        self.model = model
        self.zmuv = zmuv_transform
        self.std = StandardAudioTransform().eval()
        self.settings = SETTINGS.inference_engine
        self.context = context

        self.inference_weights = 1
        if self.settings.inference_weights:
            pad_size = context.num_labels - len(self.settings.inference_weights)
            self.inference_weights = np.pad(
                self.settings.inference_weights, (0, pad_size), "constant", constant_values=1
            )

        self.coloring = context.coloring
        self.negative_label = context.negative_label
        if self.coloring:
            self.negative_label = self.coloring.color_map[self.negative_label]

        self.sample_rate = SETTINGS.audio.sample_rate
        self.threshold = self.settings.inference_threshold
        self.inference_window_ms = self.settings.inference_window_ms
        self.smoothing_window_ms = self.settings.smoothing_window_ms
        self.tolerance_window_ms = self.settings.tolerance_window_ms
        self.sequence = self.settings.inference_sequence
        self.time_provider = time_provider
        self.reset()

    def to(self, device: torch.device):
        self.model = self.model.to(device)
        self.zmuv = self.zmuv.to(device)
        return self

    def reset(self):
        self.model.streaming_state = None
        self.curr_time = 0
        self.pred_history = []
        self.label_history = []

    def append_label(self, label: int, curr_time: float = None):
        if curr_time is None:
            curr_time = self.time_provider() * 1000
        self.label_history.append((curr_time, label))

    def sequence_present(self, curr_time: float = None) -> bool:
        if not self.sequence:
            return False
        if len(self.sequence) == 0:
            return True

        if curr_time is None:
            curr_time = self.time_provider() * 1000

        self.label_history = list(
            itertools.dropwhile(lambda x: curr_time - x[0] > self.inference_window_ms, self.label_history)
        )  # drop entries that are old

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

    def _get_prediction(self, curr_time: float) -> int:
        # drop out-dated entries
        self.pred_history = list(
            itertools.dropwhile(lambda x: curr_time - x[0] > self.smoothing_window_ms, self.pred_history)
        )
        lattice = np.vstack([t for _, t in self.pred_history])
        lattice_max = np.max(lattice, 0)
        max_label = lattice_max.argmax()
        max_prob = lattice_max[max_label]
        if self.coloring:
            max_label = self.coloring.color_map.get(max_label, self.negative_label)
        if max_prob < self.threshold:
            max_label = self.negative_label
        self.label_history.append((curr_time, max_label))
        return max_label

    def _append_probability_frame(self, p: np.ndarray, curr_time=None):
        if curr_time is None:
            curr_time = self.time_provider() * 1000
        self.pred_history.append((curr_time, p))
        return self._get_prediction(curr_time)

    def infer(self, audio_data: torch.Tensor) -> bool:
        raise NotImplementedError


class SequenceInferenceEngine(InferenceEngine):
    def __init__(self, *args):
        super().__init__(*args)
        self.blank_idx = self.context.blank_label

    @torch.no_grad()
    def infer(self, audio_data: torch.Tensor) -> bool:
        delta_ms = int(audio_data.size(-1) / self.sample_rate * 1000)
        self.std = self.std.to(audio_data.device)
        scores = self.model(
            self.zmuv(self.std(audio_data.unsqueeze(0))), None
        )  # [num_frames x 1 (batch size) x num_labels]
        scores = F.softmax(scores, -1).squeeze(1)  # [num_frames x num_labels]
        sequence_present = False
        delta_ms /= len(scores)

        for frame in scores:
            p = frame.cpu().numpy()
            p *= self.inference_weights
            p = p / p.sum()
            logging.debug(([f"{x:.3f}" for x in p.tolist()], np.argmax(p)))
            self.curr_time += delta_ms
            if np.argmax(p) == self.blank_idx:
                continue
            self._append_probability_frame(p, curr_time=self.curr_time)
            if self.sequence_present(self.curr_time):
                sequence_present = True
                break

        return sequence_present


class FrameInferenceEngine(InferenceEngine):
    def __init__(self, max_window_size_ms: int, eval_stride_size_ms: int, *args):
        super().__init__(*args)
        self.max_window_size_ms, self.eval_stride_size_ms = max_window_size_ms, eval_stride_size_ms

    @torch.no_grad()
    def infer(self, audio_data: torch.Tensor) -> bool:
        sequence_present = False
        for window in stride(audio_data, self.max_window_size_ms, self.eval_stride_size_ms, self.sample_rate):
            if window.size(-1) < 1000:
                break
            self.ingest_frame(window.squeeze(0), curr_time=self.curr_time)
            self.curr_time += self.eval_stride_size_ms
            if self.sequence_present(self.curr_time):
                sequence_present = True
                break
        return sequence_present

    @torch.no_grad()
    def ingest_frame(self, x: torch.Tensor, lengths: torch.Tensor = None, curr_time: float = None) -> int:
        self.std = self.std.to(x.device)
        if lengths is None:
            lengths = torch.tensor([x.size(-1)]).to(x.device)
        lengths = self.std.compute_lengths(lengths)
        x = self.zmuv(self.std(x.unsqueeze(0)))
        p = self.model(x, lengths).softmax(-1)[0].cpu().numpy()

        p *= self.inference_weights
        p = p / p.sum()
        label = self._append_probability_frame(p, curr_time=curr_time)
        return label
