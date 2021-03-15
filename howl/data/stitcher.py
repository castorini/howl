
from howl.data.dataset import AudioClipDataset, AudioDataset, WakeWordDataset
from howl.data.searcher import WordTranscriptSearcher
from howl.data.tokenize import Vocab
from howl.settings import SETTINGS

__all__ = ['WordStitcher']


class Stitcher:
    def __init__(self,
                 vocab: Vocab):
        self.settings = SETTINGS.inference_engine.inference_sequence
        self.vocab = vocab


class WordStitcher(Stitcher):
    def __init__(self,
                 word_searcher: WordTranscriptSearcher,
                 **kwargs):
        super().__init__(**kwargs)
        self.word_searcher = word_searcher

    def stitch(self, datasets: AudioDataset):
        raise NotImplementedError
