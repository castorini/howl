from typing import List

from howl.data.dataset.phone import PhonePhrase, PronunciationDictionary
from howl.data.tokenize import Vocab

from .base import AudioClipMetadata, FrameLabelData

__all__ = ["FrameLabeler", "WordFrameLabeler", "PhoneticFrameLabeler"]


class FrameLabeler:
    def compute_frame_labels(self, metadata: AudioClipMetadata) -> FrameLabelData:
        raise NotImplementedError


class PhoneticFrameLabeler(FrameLabeler):
    def __init__(self, pronounce_dict: PronunciationDictionary, phrases: List[PhonePhrase]):
        self.pronounce_dict = pronounce_dict
        self.phrases = phrases

    def compute_frame_labels(self, metadata: AudioClipMetadata) -> FrameLabelData:
        frame_labels = dict()
        start_timestamp = []
        char_indices = []

        start = 0
        # TODO:: must be pronounciation instead of the transcription
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

        return FrameLabelData(frame_labels, start_timestamp, char_indices)


class WordFrameLabeler(FrameLabeler):
    def __init__(self, vocab: Vocab, ceil_word_boundary: bool = False):
        self.vocab = vocab
        self.ceil_word_boundary = ceil_word_boundary

    def compute_frame_labels(self, metadata: AudioClipMetadata) -> FrameLabelData:
        frame_labels = dict()
        start_timestamp = []
        char_indices = []

        char_idx = 0
        for word in metadata.transcription.split():
            vocab_found, remaining_transcript = self.vocab.trie.max_split(word)
            word_size = len(word.rstrip())

            # if the current word is in vocab, store necessary informations
            if vocab_found and remaining_transcript == "":
                label = self.vocab[word]
                end_timestamp = metadata.end_timestamps[char_idx + word_size - 1]
                frame_labels[end_timestamp] = label
                char_indices.append((label, list(range(char_idx, char_idx + word_size))))
                start_timestamp.append((label, metadata.end_timestamps[char_idx - 1] if char_idx > 0 else 0.0))

            char_idx += word_size + 1  # space

        return FrameLabelData(frame_labels, start_timestamp, char_indices)
