from collections import defaultdict
from typing import List
import itertools
import logging
import re
import time

from pydantic import BaseSettings
import numpy as np
import torch
import torch.nn as nn

from howl.data.dataset import PhonePhrase
from howl.data.transform import ZmuvTransform, StandardAudioTransform


__all__ = ['InferenceEngine',
           'InferenceEngineSettings',
           'PhoneticTranscriptSearcher',
           'TranscriptSearcher',
           'WordTranscriptSearcher',
           'LabelColoring']


class LabelColoring:
    def __init__(self):
        self.color_map = {}
        self.color_counter = 0
        self.label_counter = 0

    def append_label(self, label: int, color: int = None):
        color = self._inc_color_counter(color)
        self.color_map[label] = color

    def _inc_color_counter(self, color: int = None):
        if color is None:
            color = self.color_counter
        else:
            self.color_counter = max(self.color_counter, color + 1)
        self.color_counter += 1
        return color

    def extend_sequence(self, size: int, color: int = None):
        color = self._inc_color_counter(color)
        for label in range(self.label_counter, self.label_counter + size):
            self.color_map[label] = color
        self.label_counter += size

    @classmethod
    def sequential_coloring(cls, num_labels: int):
        coloring = cls()
        for label_idx in range(num_labels):
            coloring.append_label(label_idx)
        return coloring


class InferenceEngineSettings(BaseSettings):
    inference_weights: List[float] = None
    inference_sequence: List[int] = [0]
    inference_window_ms: float = 2000  # look at last of these seconds
    smoothing_window_ms: float = 50  # prediction smoothed
    tolerance_window_ms: float = 500  # negative label between words
    inference_threshold: float = 0  # positive label probability must rise above this threshold


class TranscriptSearcher:
    def __init__(self, settings: InferenceEngineSettings = InferenceEngineSettings()):
        self.settings = settings

    def search(self, item: str) -> bool:
        raise NotImplementedError


class WordTranscriptSearcher(TranscriptSearcher):
    def __init__(self, vocab: List[str], **kwargs):
        super().__init__(**kwargs)
        self.vocab = vocab
        self.wakeword = ' '.join(np.array(self.vocab)[self.settings.inference_sequence])

    def search(self, item: str) -> bool:
        return self.wakeword in item


class PhoneticTranscriptSearcher(TranscriptSearcher):
    def __init__(self, phrases: List[PhonePhrase], coloring: LabelColoring, **kwargs):
        super().__init__(**kwargs)
        self.phrases = phrases
        self.coloring = coloring
        label_map = [(phrase.audible_transcript, coloring.color_map[idx]) for idx, phrase in enumerate(phrases)]
        buckets = defaultdict(list)
        for transcript, color in label_map:
            buckets[color].append(transcript)
        pattern_strings = []
        for _, transcripts in sorted(buckets.items(), key=lambda x: x[0]):
            pattern_strings.append('(' + '|'.join(f'({x})' for x in transcripts) + ')')
        pattern_strings = np.array(pattern_strings)[self.settings.inference_sequence]
        pattern_str = '^.*' + ' '.join(pattern_strings) + '.*$'
        logging.info(f'Using search pattern {pattern_str}')
        self.pattern = re.compile(pattern_str)

    def search(self, item: str) -> bool:
        transcript = PhonePhrase.from_string(item).audible_transcript
        return self.pattern.match(transcript) is not None


class InferenceEngine:
    def __init__(self,
                 model: nn.Module,
                 zmuv_transform: ZmuvTransform,
                 negative_label: int,
                 settings: InferenceEngineSettings = InferenceEngineSettings(),
                 coloring: LabelColoring = None,
                 time_provider=time.time):
        self.model = model
        self.zmuv = zmuv_transform
        self.std = StandardAudioTransform().eval()
        self.settings = settings
        inference_weights = 1 if settings.inference_weights is None else np.array(settings.inference_weights)
        self.inference_weights = inference_weights
        self.negative_label = negative_label
        self.threshold = settings.inference_threshold
        self.inference_window_ms = settings.inference_window_ms
        self.smoothing_window_ms = settings.smoothing_window_ms
        self.tolerance_window_ms = settings.tolerance_window_ms
        self.sequence = settings.inference_sequence
        self.coloring = coloring
        self.time_provider = time_provider
        self.reset()

    def reset(self):
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
        lattice_max = np.max(lattice, 0)
        max_label = lattice_max.argmax()
        max_prob = lattice_max[max_label]
        if self.coloring:
            max_label = self.coloring.color_map.get(max_label, self.negative_label)
        if max_prob < self.threshold:
            max_label = self.negative_label
        self.label_history.append((curr_time, max_label))
        return max_label

    @torch.no_grad()
    def infer(self,
              x: torch.Tensor,
              lengths: torch.Tensor = None,
              curr_time: float = None) -> int:

        if curr_time is None:
            curr_time = self.time_provider() * 1000

        self.std = self.std.to(x.device)
        if lengths is None:
            lengths = torch.tensor([x.size(-1)]).to(x.device)
        lengths = self.std.compute_lengths(lengths)
        x = self.zmuv(self.std(x.unsqueeze(0)))
        p = self.model(x, lengths).softmax(-1)[0].cpu().numpy()

        p *= self.inference_weights
        p = p / p.sum()

        self.pred_history.append((curr_time, p))
        label = self._get_prediction(curr_time)
        logging.debug(([f'{x:.3f}' for x in p.tolist()], label))
        return label
