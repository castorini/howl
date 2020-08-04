import numpy as np

from .base import AlignedTranscription, Aligner


__all__ = ['StubAligner']


class StubAligner(Aligner):
    def align(self, audio) -> AlignedTranscription:
        start = 0
        end = audio.audio_data.size(0) / audio.sample_rate * 1000
        transcription = audio.metadata.transcription.lower()
        return AlignedTranscription(transcription=transcription,
                                    end_timestamps=np.linspace(start, end, len(transcription)).tolist())
