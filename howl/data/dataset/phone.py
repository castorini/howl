import enum
from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import List, Mapping

from howl.settings import SETTINGS

__all__ = ["PhoneEnum", "PronunciationDictionary", "PhonePhrase"]


class PhoneEnum(enum.Enum):
    SILENCE = "sil"
    SILENCE_OPTIONAL = "sp"
    SPEECH_UNKNOWN = "spn"


@dataclass
class Phone:
    text: str

    def __post_init__(self):
        self.text = self.text.lower().strip()
        self.is_speech = self.text not in (
            PhoneEnum.SILENCE.value,
            PhoneEnum.SILENCE_OPTIONAL.value,
            PhoneEnum.SPEECH_UNKNOWN.value,
        )

    def __str__(self):
        return self.text

    def __eq__(self, other: "Phone"):
        return other.text == self.text


@dataclass
class PhonePhrase:
    phones: List[Phone]

    def __post_init__(self):
        self.audible_phones = [x for x in self.phones if x.is_speech]
        self.audible_transcript = " ".join(x.text for x in self.audible_phones)
        self.sil_indices = [idx for idx, x in enumerate(self.phones) if not x.is_speech]

    @property
    def text(self):
        return str(self)

    @classmethod
    def from_string(cls, string: str):
        return cls([Phone(x) for x in string.split()])

    def __str__(self):
        return " ".join(x.text for x in self.phones)

    def all_idx_to_transcript_idx(self, phone_idx: int) -> int:
        if phone_idx >= len(self.phones):
            raise ValueError(f"Given phone idx ({phone_idx}) is greater than the number of phones ({len(self.phones)})")
        all_idx_without_space = sum(map(len, [phone.text for phone in self.phones[: phone_idx + 1]]))
        return all_idx_without_space + phone_idx  # add phone_idx for spaces between phones

    def audible_idx_to_all_idx(self, audible_idx: int) -> int:
        if audible_idx >= len(self.audible_phones):
            raise ValueError(
                f"Given audible phone idx ({audible_idx}) is greater than"
                "the number of audible phones ({len(self.audible_phones)})"
            )
        offset = 0
        for sil_idx in self.sil_indices:
            if sil_idx <= audible_idx + offset:
                offset += 1
        return offset + audible_idx

    def audible_index(self, item: "PhonePhrase", start: int = 0):
        item_len = len(item.audible_phones)
        if item_len == 0:
            raise ValueError(f"query phrase has empty audible_phones: {item.audible_transcript}")
        self_len = len(self.audible_phones)
        for idx in range(start, self_len - item_len + 1):
            if all(x == y for x, y in zip(item.audible_phones, self.audible_phones[idx : idx + item_len])):
                return idx
        raise ValueError(f"query phrase is not found: {item.audible_transcript}")


class PronunciationDictionary:
    def __init__(self, data_dict: Mapping[str, List[PhonePhrase]]):
        self.word2phone = data_dict

    @lru_cache(maxsize=SETTINGS.cache.cache_size)
    def __contains__(self, key: str):
        return key.strip().lower() in self.word2phone

    @lru_cache(maxsize=SETTINGS.cache.cache_size)
    def encode(self, word: str) -> List[PhonePhrase]:
        word = word.strip().lower()
        if word not in self.word2phone:
            raise ValueError(f"word is not in the dictionary: {word}")
        return self.word2phone[word.strip().lower()]

    @classmethod
    def from_file(cls, filename: Path):
        data = defaultdict(list)
        with filename.open() as f:
            for line in f:
                if line.startswith(";"):
                    continue
                word, pronunciation = line.split(" ", 1)
                if len(word) == 0 or len(pronunciation) == 0:
                    continue
                data[word.lower()].append(PhonePhrase.from_string(pronunciation.strip().lower()))
        return cls(data)
