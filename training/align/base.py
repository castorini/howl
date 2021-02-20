from typing import List

from pydantic import BaseModel


__all__ = ['AlignedTranscription', 'Aligner']


class AlignedTranscription(BaseModel):
    transcription: str
    end_timestamps: List[float]


class Aligner:
    def align(self, audio) -> AlignedTranscription:
        from howl.data.dataset import AudioClipExample
        audio  # type: AudioClipExample
        raise NotImplementedError

