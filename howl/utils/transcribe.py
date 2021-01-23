from pathlib import Path
import os

from pocketsphinx import AudioFile, get_model_path


class SpeechToText():
    def __init__(self):
        model_path = get_model_path()
        self.config = {
            'verbose': False,
            'buffer_size': 2048,
            'no_search': False,
            'full_utt': False,
            'hmm': os.path.join(model_path, 'en-us'),
            'lm': os.path.join(model_path, 'en-us.lm.bin'),
            'dict': os.path.join(model_path, 'cmudict-en-us.dict')
        }

    def transcribe(self, audio_file: Path):
        self.config['audio_file'] = audio_file
        transcription = ''
        for phase in AudioFile(**self.config):
            transcription = str(phase)
            break
        return transcription
