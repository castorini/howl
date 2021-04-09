import os

from pocketsphinx import AudioFile


class SphinxKeywordDetector():
    def __init__(self, target_transcription, threshold=1e-20, verbose=False):
        self.target_transcription = target_transcription
        self.verbose = verbose
        self.kws_config = {
            'verbose': self.verbose,
            'keyphrase': self.target_transcription,
            'kws_threshold': threshold,
            'lm': False,
        }

    def detect(self, file_name):

        kws_results = []

        self.kws_config['audio_file'] = file_name
        audio = AudioFile(**self.kws_config)

        for phrase in audio:
            result = phrase.segments(detailed=True)

            # TODO:: confirm that when multiple keywords are detected, every detection is valid
            if len(result) == 1:
                start_time = result[0][2] * 10
                end_time = result[0][3] * 10
                if self.verbose:
                    print('%4sms ~ %4sms' % (start_time, end_time))
                kws_results.append((start_time, end_time))

        return kws_results
