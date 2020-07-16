from typing import List

from pydantic import BaseModel


__all__ = ['AlignedTranscription']


class AlignedTranscription(BaseModel):
    transcription: str
    end_timestamps: List[float]
