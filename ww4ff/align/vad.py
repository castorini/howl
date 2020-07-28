import numpy as np
import webrtcvad

from .base import AlignedTranscription, Aligner


__all__ = ['LeftRightVadAligner']


class LeftRightVadAligner(Aligner):
    def __init__(self, frame_ms: int = 30):
        self.frame_ms = frame_ms

    def align(self, audio) -> AlignedTranscription:
        def detect_vad(frames):
            start = 0
            vad = webrtcvad.Vad(3)
            for frame in frames:
                buf = frame.numpy() * 32767
                buf = buf.astype(np.int16).tobytes()
                if frame.size(0) < frame_len or vad.is_speech(buf, audio.sample_rate):
                    break
                start += self.frame_ms
            return start
        transcription = audio.metadata.transcription.lower()
        frame_len = int(self.frame_ms / 1000 * audio.sample_rate)
        frames = audio.audio_data.split(frame_len)
        length = int(1000 * audio.audio_data.size(0) / audio.sample_rate)
        start = detect_vad(frames)
        frames = audio.audio_data.flip(0).split(frame_len)
        end = length - detect_vad(frames)
        if end <= start:
            start = 0
            end = length
        return AlignedTranscription(transcription=transcription,
                                    end_timestamps=np.linspace(start, end, len(transcription)).tolist())
