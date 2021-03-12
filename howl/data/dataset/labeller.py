
from dataclasses import dataclass
from typing import List

from howl.data.dataset.phone import PhonePhrase

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
    def __init__(self, words: List[str], ceil_word_boundary: bool = False):
        self.words = words
        self.ceil_word_boundary = ceil_word_boundary

    def compute_frame_labels(self, metadata: AudioClipMetadata) -> FrameLabelData:
        frame_labels = dict()
        t = f' {metadata.transcription} '
        start = 0
        for idx, word in enumerate(self.words):
            while True:
                try:
                    start = t.index(word, start)
                except ValueError:
                    break
                while self.ceil_word_boundary and start + len(word) < len(t) - 1 and t[start + len(word)] != ' ':
                    start += 1
                frame_labels[metadata.end_timestamps[start + len(word.rstrip()) - 2]] = idx
                start += 1
        return FrameLabelData(frame_labels)
