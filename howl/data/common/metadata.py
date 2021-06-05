from pathlib import Path
from typing import List, Optional

from pydantic import BaseModel

from howl.data.common.phone import Phone, PhonePhrase

__all__ = ["AudioClipMetadata", "UNKNOWN_TRANSCRIPTION", "NEGATIVE_CLASS"]


UNKNOWN_TRANSCRIPTION = "[UNKNOWN]"
NEGATIVE_CLASS = "[NEGATIVE]"


class AudioClipMetadata(BaseModel):
    path: Optional[Path] = Path(".")
    phone_strings: Optional[List[str]]
    words: Optional[List[str]]
    phone_end_timestamps: Optional[List[float]]
    end_timestamps: Optional[List[float]]  # TODO: remove, backwards compat right now
    transcription: Optional[str] = ""

    # TODO:: id should be an explicit variable in order to support datasets creation with the audio data in memory
    @property
    def audio_id(self) -> str:
        return self.path.name.split(".", 1)[0]

    @property
    def phone_phrase(self) -> Optional[PhonePhrase]:
        return PhonePhrase([Phone(x) for x in self.phone_strings]) if self.phone_strings else None
