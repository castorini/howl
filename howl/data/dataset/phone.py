from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, List
import enum


__all__ = ['PhoneEnum',
           'PronunciationDictionary',
           'PhonePhrase']


class PhoneEnum(enum.Enum):
    SILENCE = 'sil'
    SILENCE_OPTIONAL = 'sp'
    SPEECH_UNKNOWN = 'spn'


@dataclass
class Phone:
    text: str

    def __post_init__(self):
        self.text = self.text.lower().strip()
        self.is_speech = self.text not in (PhoneEnum.SILENCE.value,
                                           PhoneEnum.SILENCE_OPTIONAL.value,
                                           PhoneEnum.SPEECH_UNKNOWN.value)

    def __str__(self):
        return self.text

    def __eq__(self, other: 'Phone'):
        return other.text == self.text


@dataclass
class PhonePhrase:
    phones: List[Phone]

    def __post_init__(self):
        self.audible_phones = [x for x in self.phones if x.is_speech]
        self.audible_transcript = ' '.join(x.text for x in self.audible_phones)
        self.sil_indices = [idx for idx, x in enumerate(self.phones) if not x.is_speech]

    @property
    def text(self):
        return str(self)

    @classmethod
    def from_string(cls, string: str):
        return cls([Phone(x) for x in string.split()])

    def __str__(self):
        return ' '.join(x.text for x in self.phones)

    def all_idx_to_transcript_idx(self, index: int) -> int:
        return sum(map(len, [phone.text for phone in self.phones[:index]])) + index

    def audible_idx_to_all_idx(self, index: int) -> int:
        offset = 0
        for sil_idx in self.sil_indices:
            if sil_idx <= index + offset:
                offset += 1
        return offset + index

    def audible_index(self, item: 'PhonePhrase', start: int = 0):
        item_len = len(item.audible_phones)
        self_len = len(self.audible_phones)
        for idx in range(start, self_len - item_len + 1):
            if all(x == y for x, y in zip(item.audible_phones, self.audible_phones[idx:idx + item_len])):
                return idx
        raise ValueError


class PronunciationDictionary:
    def __init__(self, data_dict: Mapping[str, List[PhonePhrase]]):
        self.word2phone = data_dict

    def encode(self, word: str) -> List[PhonePhrase]:
        return self.word2phone[word.lower().strip()]

    @classmethod
    def from_file(cls, filename: Path):
        data = defaultdict(list)
        with filename.open() as f:
            for line in f:
                if line.startswith(';'):
                    continue
                try:
                    word, pronunciation = line.split(' ', 1)
                except:
                    pass
                data[word.lower()].append(PhonePhrase(list(map(Phone, pronunciation.strip().lower().split()))))
        return cls(data)
