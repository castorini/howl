import random
from typing import Sequence

import torch

from howl.data.common.batch import ClassificationBatch, SequenceBatch
from howl.data.common.example import WakeWordClipExample
from howl.data.common.tokenizer import TranscriptTokenizer
from howl.data.transform.operator import pad, random_slice, tensorize_audio_data

__all__ = ["WakeWordFrameBatchifier", "AudioSequenceBatchifier"]


class AudioSequenceBatchifier:
    def __init__(self, negative_label: int, tokenizer: TranscriptTokenizer, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        self.tokenizer = tokenizer
        self.negative_label = negative_label

    def __call__(self, examples: Sequence[WakeWordClipExample]) -> SequenceBatch:
        audio_data_lst = []
        labels_lst = []
        for ex in examples:
            labels = self.tokenizer.encode(ex.metadata.transcription)
            labels_lst.append(labels)
            audio_data_lst.append(ex.audio_data)
        audio_data_lengths = [audio_data.size(-1) for audio_data in audio_data_lst]
        labels_lengths = list(map(len, labels_lst))
        audio_tensor, data = tensorize_audio_data(
            audio_data_lst, labels_lst=labels_lst, labels_lengths=labels_lengths, input_lengths=audio_data_lengths
        )
        labels_lst = torch.tensor(pad(data["labels_lst"], element=self.negative_label))
        labels_lengths = torch.tensor(data["labels_lengths"])
        return SequenceBatch(audio_tensor, labels_lst, torch.tensor(data["input_lengths"]), labels_lengths)


class WakeWordFrameBatchifier:
    def __init__(
        self,
        negative_label: int,
        positive_sample_prob: float = 0.5,
        window_size_ms: int = 500,
        sample_rate: int = 16000,
        positive_delta_ms: int = 150,
        eps_ms: int = 20,
        pad_to_window: bool = True,
    ):
        self.positive_sample_prob = positive_sample_prob
        self.window_size_ms = window_size_ms
        self.sample_rate = sample_rate
        self.negative_label = negative_label
        self.positive_delta_ms = positive_delta_ms
        self.eps_ms = eps_ms
        self.pad_to_window = pad_to_window

    def __call__(self, examples: Sequence[WakeWordClipExample]) -> ClassificationBatch:
        new_examples = []
        for ex in examples:
            # if the sample does not contain any positive label
            if not ex.label_data.timestamp_label_map:
                new_examples.append(
                    (self.negative_label, random_slice([ex], int(self.sample_rate * self.window_size_ms / 1000))[0])
                )
                continue

            select_negative = random.random() > self.positive_sample_prob

            # pick a random positive word to genrate a positive sample
            if not select_negative:
                end_ms, label = random.choice(list(ex.label_data.timestamp_label_map.items()))
                end_ms_rand = end_ms + (random.random() * self.eps_ms)
                b = int((end_ms_rand / 1000) * self.sample_rate)
                a = max(b - int((self.window_size_ms / 1000) * self.sample_rate), 0)
                if random.random() < 0:
                    closest_ms = min(
                        filter(lambda k: end_ms - k > 0, ex.label_data.timestamp_label_map.keys()),
                        key=lambda k: end_ms - k,
                        default=-1,
                    )
                    if closest_ms >= 0:
                        a = max(a, int((closest_ms / 1000) * self.sample_rate))
                if b - a < 0:
                    select_negative = True
                else:
                    new_examples.append((label, ex.emplaced_audio_data(ex.audio_data[..., a:b])))

            # use the interval of negative labels to generate a negative sample from audio with positive label
            if select_negative:
                positive_intervals = [
                    (v - self.positive_delta_ms, v + self.positive_delta_ms)
                    for v in ex.label_data.timestamp_label_map.values()
                ]
                positive_intervals = sorted(positive_intervals, key=lambda x: x[0])
                negative_intervals = []
                last_positive = 0
                for a, b in positive_intervals:
                    if last_positive < a:
                        negative_intervals.append((last_positive, a))
                    last_positive = b
                negative_intervals.append((b, int(len(ex.audio_data) / 16000 * 1000)))
                a, b = random.choice(negative_intervals)
                if b - a > self.window_size_ms:
                    a = random.randint(0, int(b - self.window_size_ms))
                    b = a + self.window_size_ms
                new_examples.append((self.negative_label, ex.emplaced_audio_data(ex.audio_data[..., a:b])))

        labels_lst = [x[0] for x in new_examples]
        max_length = int(self.window_size_ms / 1000 * self.sample_rate) if self.pad_to_window else None
        audio_tensor, extra_data = tensorize_audio_data(
            [x[1].audio_data for x in new_examples],
            rand_append=True,
            max_length=max_length,
            labels_lst=labels_lst,
            lengths=[x[1].audio_data.size(-1) for x in new_examples],
        )
        labels_tensor = torch.tensor(extra_data["labels_lst"])
        lengths = torch.tensor(extra_data["lengths"])
        return ClassificationBatch(audio_tensor, labels_tensor, lengths)
