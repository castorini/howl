
from dataclasses import dataclass
from typing import List

from howl.data.dataset.phone import PhonePhrase
from howl.data.tokenize import Vocab

from .base import AudioClipMetadata, FrameLabelData

__all__ = ['FrameLabeler',
           'WordFrameLabeler',
           'PhoneticFrameLabeler']


class FrameLabeler:
    def compute_frame_labels(self, metadata: AudioClipMetadata) -> FrameLabelData:
        raise NotImplementedError


class PhoneticFrameLabeler(FrameLabeler):
    def __init__(self, phrases: List[PhonePhrase]):
        self.phrases = phrases

    def compute_frame_labels(self, metadata: AudioClipMetadata) -> FrameLabelData:
        frame_labels = dict()
        start = 0
        pp = PhonePhrase.from_string(metadata.transcription)
        for idx, phrase in enumerate(self.phrases):
            while True:
                try:
                    start = pp.audible_index(phrase, start)
                except ValueError:
                    break
                # TODO: make alignment to token instead of character
                start = pp.all_idx_to_transcript_idx(pp.audible_idx_to_all_idx(start))
                frame_labels[metadata.end_timestamps[start + len(str(phrase)) - 1]] = idx
                start += 1
        return FrameLabelData(frame_labels)


class WordFrameLabeler(FrameLabeler):
    def __init__(self, vocab: Vocab, ceil_word_boundary: bool = False):
        self.vocab = vocab
        self.ceil_word_boundary = ceil_word_boundary

    def compute_frame_labels(self, metadata: AudioClipMetadata) -> FrameLabelData:
        frame_labels = dict()
        # TODO:: consider iterating over end_timestamps instead of every character
        t = f' {metadata.transcription} '
        char_idx = 0
        for label in range(len(self.vocab)):
            word = self.vocab[label]
            while True:
                try:
                    # check if transcript contains the current word
                    char_idx = t.index(word, char_idx)
                except ValueError:
                    break
                # capture the full word if ceil_word_boundary is true
                while self.ceil_word_boundary and char_idx + len(word) < len(t) - 1 and t[char_idx + len(word)] != ' ':
                    char_idx += 1
                # record the ending timestamp with corresponding label
                frame_labels[metadata.end_timestamps[char_idx + len(word.rstrip()) - 2]] = label
                char_idx += 1
        return FrameLabelData(frame_labels)
