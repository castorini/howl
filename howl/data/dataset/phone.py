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
        """get index based on transcription for the given phone_idx

        pp = PhonePhrase.from_string("abc def ghi")
        pp.all_idx_to_transcript_idx(0) # 3 - where the first phone (abc) finishes
        pp.all_idx_to_transcript_idx(1) # 7 - where the second phone (def) finishes
        pp.all_idx_to_transcript_idx(2) # 11 - where the third phone (ghi) finishes

        Args:
            phone_idx (int): target phone idx

        Raises:
            ValueError: if phone idx is out of bound

        Returns:
            int: transcription idx where the phone at the given phone_idx finishes
        """
        if phone_idx >= len(self.phones):
            raise ValueError(f"Given phone idx ({phone_idx}) is greater than the number of phones ({len(self.phones)})")
        all_idx_without_space = sum(map(len, [phone.text for phone in self.phones[: phone_idx + 1]]))
        return all_idx_without_space + phone_idx  # add phone_idx for spaces between phones

    def audible_idx_to_all_idx(self, audible_idx: int) -> int:
        """convert given audible index to phone index including all non speech phones

        pp = PhonePhrase.from_string("abc sil ghi")
        pp.audible_idx_to_all_idx(0) # 0 - where the first audible phone (abc) is located in the whole phrase
        pp.audible_idx_to_all_idx(1) # 2 - where the second audible phone (abc) is located in the whole phrase

        Args:
            audible_idx (int): audible phone index to convert

        Raises:
            ValueError: if audible phone index is out of bound

        Returns:
            int: the index of the audible phone in the whole phrase
        """
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

    def audible_index(self, query: "PhonePhrase", start: int = 0) -> int:
        """find the starting audible index of the given phrase in the current phrase

        pp = PhonePhrase.from_string("abc sil ghi")
        ghi_pp = PhonePhrase.from_string("ghi")
        pp.audible_index(ghi_pp, 0) # 1 - audible index of the query phone (ghi)

        Args:
            query (PhonePhrase): phone phrase to be searched
            start (int, optional): starting index in the whole phrase. Defaults to 0.

        Raises:
            ValueError: when the query phone phrase does not contain any phone
            ValueError: when the query phone phrase is not found

        Returns:
            int: audible index of the query phone phase
        """
        query_len = len(query.audible_phones)
        if query_len == 0:
            raise ValueError(f"query phrase has empty audible_phones: {query.audible_transcript}")
        self_len = len(self.audible_phones)
        for idx in range(start, self_len - query_len + 1):
            if all(x == y for x, y in zip(query.audible_phones, self.audible_phones[idx : idx + query_len])):
                return idx
        raise ValueError(f"query phrase is not found: {query.audible_transcript}")


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
