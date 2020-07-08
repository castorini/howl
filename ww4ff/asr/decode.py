from typing import List

import numpy as np

from .deepspeech import AsrOutput
from .align import AlignedTranscription


__all__ = ['DeepSpeechAligner']


class DeepSpeechAligner:
    def __init__(self,
                 alphabet: str = ' abcdefghijklmnopqrstuvwxyz\''):
        self.alphabet = alphabet
        self.alphabet_set = set(alphabet)

    def align(self, outputs: List[AsrOutput], transcription: str) -> AlignedTranscription:
        transcription = transcription.lower()
        transcription = list(filter(lambda x: x in self.alphabet_set, transcription))
        labels = [self.alphabet.index(x) for x in transcription]
        all_probs = []
        intervals = []
        for output in outputs:
            probs = output.probs
            start = output.start_ms
            end = output.end_ms
            all_probs.append(probs)
            intervals.append((probs.shape[0], start, end))
        probs = np.concatenate(all_probs, 0)
        scores = ctc_forward(probs, labels)
        alignment = compute_simple_max_alignment(scores)
        align_end_ms_list = []
        curr_idx = 0
        for length, start, end in intervals:
            interval_ms = (end - start) / length
            align_end_ms_list.extend((alignment[curr_idx:curr_idx + length] * interval_ms + start).tolist())
            curr_idx += length
        return AlignedTranscription(transcription=''.join(transcription), end_timestamps=align_end_ms_list)


def compute_simple_max_alignment(scores: np.ndarray):
    scores = scores[:, 1::2] + scores[:, 2::2]
    return np.argmax(scores, 0)


def ctc_forward(probs2d: np.ndarray, labels: List[int]):
    def compute_alpha_bar(t, s):
        return compute_alpha(t - 1, s) / norms[t - 1] + compute_alpha(t - 1, s - 1) / norms[t - 1]

    def compute_alpha(t, s):
        if s < len(labels_prime) - 2 * (probs2d.shape[0] - t):
            computed_table[t, s] = True
            return 0
        try:
            if computed_table[t, s]:
                return scores[t, s]
            if labels_prime[s] == blank_idx or labels_prime[s - 2] == labels_prime[s]:
                scores[t, s] = compute_alpha_bar(t, s) * probs2d[t, labels_prime[s]]
            else:
                scores[t, s] = (compute_alpha_bar(t, s) + compute_alpha(t - 1, s - 2) / norms[t - 1]) * probs2d[t, labels_prime[s]]
            computed_table[t, s] = True
            return scores[t, s]
        except:
            return 0
    blank_idx = probs2d.shape[1] - 1
    labels_prime = [blank_idx]
    for label in labels:
        labels_prime.append(label)
        labels_prime.append(blank_idx)
    scores = np.zeros((probs2d.shape[0], len(labels_prime)))
    computed_table = np.zeros_like(scores, dtype=np.bool)
    norms = np.zeros(probs2d.shape[0])
    computed_table[0, :] = True
    scores[0, 0] = probs2d[0, -1]
    scores[0, 1] = probs2d[0, labels[0]]
    norms[0] = scores[0].sum()
    for t in range(1, probs2d.shape[0]):
        for s in range(len(labels_prime)):
            compute_alpha(t, s)
        norms[t] = scores[t].sum()
    return scores
