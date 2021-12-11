import numpy as np

from .base import AlignedTranscription, Aligner


class StubAligner(Aligner):
    """Generate transcription with stub alignments"""

    def align(self, audio) -> AlignedTranscription:
        """Generate alignment for given audio by evenly splitting the characters of transcription"""
        start = 0
        end = audio.audio_data.size(0) / audio.sample_rate * 1000
        transcription = audio.metadata.transcription.lower()
        return AlignedTranscription(
            transcription=transcription, end_timestamps=np.linspace(start, end, len(transcription)).tolist()
        )
