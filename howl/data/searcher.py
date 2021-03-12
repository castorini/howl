import logging
import re
from collections import defaultdict
from typing import List

import numpy as np
from howl.data.dataset.phone import PhonePhrase
from howl.data.tokenize import Vocab, WakeWordTokenizer
from howl.settings import SETTINGS

__all__ = ['LabelColoring',
           'PhoneticTranscriptSearcher',
           'TranscriptSearcher',
           'WordTranscriptSearcher']


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


class TranscriptSearcher:
    def __init__(self):
        self.settings = SETTINGS.inference_engine

    def search(self, item: str) -> bool:
        raise NotImplementedError

    def contains_any(self, item: str) -> bool:
        raise NotImplementedError


class WordTranscriptSearcher(TranscriptSearcher):
    # Utilize Vocab class
    def __init__(self, vocab: Vocab, **kwargs):
        super().__init__(**kwargs)
        self.vocab = vocab
        self.tokenizer = WakeWordTokenizer(self.vocab, False)
        self.inference_sequence_str = ''.join(map(str, self.settings.inference_sequence))
        # self.wakeword = self.vocab.decode(self.settings.inference_sequence)

    def search(self, item: str) -> bool:
        """return true if wakeword is in the item
        Args:
            item (str): string to search from
        Returns:
            bool: true if wakeword is in the item
        """
        encoded_output = self.tokenizer.encode(item)
        encoded_str = ''.join(map(str, encoded_output))
        return self.inference_sequence_str in encoded_str

    def contains_any(self, item: str) -> bool:
        """retrun true if at least one vocab is in the item
        Args:
            item (str): string to search from
        Returns:
            bool: true if at least one vocab is in the item
        """
        encoded_output = self.tokenizer.encode(item)
        return any((encoding != self.vocab.oov_token_id) for encoding in encoded_output)

    def count_vocab(self, item: str, ignore_oov: bool = True) -> dict:
        """generate counter per vocab for the item
        Args:
            item (str): string to analyze
            ignore_oov (bool, optional): set to true to ignore oov word. Defaults to True.
        Returns:
            dict[str, int]: number of occurance in the item for each vocab
        """
        encoded_output = self.tokenizer.encode(item)
        counter = dict((self.vocab[i], 0) for i in range(len(self.vocab)))
        for encoding in encoded_output:
            if ignore_oov and encoding == self.vocab.oov_token_id:
                continue
            counter[self.vocab[encoding]] += 1

        return counter


class PhoneticTranscriptSearcher(TranscriptSearcher):
    def __init__(self, phrases: List[PhonePhrase], coloring: LabelColoring, **kwargs):
        super().__init__(**kwargs)
        self.phrases = phrases
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

    def contains_any(self, item: str) -> bool:
        transcript = PhonePhrase.from_string(item).audible_transcript
        return any(word.audible_transcript in transcript for word in self.phrases)
