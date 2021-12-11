from enum import Enum, unique
from typing import List

from howl.data.common.vocab import Vocab

__all__ = ["WakeWordTokenizer", "TranscriptTokenizer"]


@unique
class TokenType(str, Enum):
    """String based Enum for token type"""

    PHONE = "phone"
    WORD = "word"


class TranscriptTokenizer:
    """Handle conversion between transcription and encoded ids"""

    def encode(self, transcript: str) -> List[int]:
        """Generate token ids for each entry in the transcript"""
        raise NotImplementedError

    def decode(self, ids: List[int]) -> str:
        """Generate transcription for the given list of ids"""
        raise NotImplementedError


class WakeWordTokenizer(TranscriptTokenizer):
    """Handle conversion between transcription and encoded ids based on word-level vocabulary"""

    # Only used for ctc objective
    def __init__(self, vocab: Vocab, ignore_oov: bool = True):
        self.vocab = vocab
        self.ignore_oov = ignore_oov

    def encode(self, transcript: str) -> List[int]:
        """Generate token ids for each word in the transcript"""
        encoded_output = []

        for word in transcript.lower().split():
            vocab_found, remaining_transcript = self.vocab.trie.max_split(word)

            # append corresponding label
            if vocab_found and remaining_transcript == "":
                # word exists in the vocab
                encoded_output.append(self.vocab[word])
            elif not self.ignore_oov:
                # label oov word
                if self.vocab.oov_token_id is None:
                    raise ValueError("label for oov word is not specified")
                encoded_output.append(self.vocab.oov_token_id)

        return encoded_output

    def decode(self, ids: List[int]) -> str:
        """Generate transcription for the given list of ids"""
        return " ".join(self.vocab[id] for id in ids)
