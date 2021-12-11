import numpy as np
from textgrids import TextGrid

from .base import AlignedTranscription


class MfaTextGridConverter:
    """Converts MFA alignment (TextGrid) into AlginedTranscription"""

    def __init__(self, split_by_char: bool = True, use_phones: bool = False):
        assert split_by_char, "word-level not implemented"
        self.split_by_char = split_by_char
        self.use_phones = use_phones

    def convert(self, text_grid: TextGrid) -> AlignedTranscription:
        """Generate AlignedTranscription from TextGrid file created by MFA"""
        end_timestamps = []
        words = []
        key = "phones" if self.use_phones else "words"
        if self.split_by_char:
            for word in text_grid[key]:
                word_len = len(word.text)
                if word_len == 0:
                    continue
                start_timestamp, end_timestamp = 1000 * word.xmin, 1000 * word.xmax
                interval = np.linspace(start_timestamp, end_timestamp, word_len)
                end_timestamps.extend(interval.tolist())
                words.append(word.text)
                end_timestamps.append(end_timestamp)  # spaces
        if len(end_timestamps) > 0:
            end_timestamps.pop()  # pop last space
        transcript = " ".join(words)
        assert len(transcript) == len(end_timestamps), "unequal alignment"
        return AlignedTranscription(transcription=transcript.lower(), end_timestamps=end_timestamps)
