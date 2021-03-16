
import itertools
from pathlib import Path

import soundfile
import torch
from howl.data.dataset import AudioClipDataset, AudioDataset, WakeWordDataset
from howl.data.searcher import WordTranscriptSearcher
from howl.data.tokenize import Vocab
from howl.settings import SETTINGS

__all__ = ['WordStitcher']


class Stitcher:
    def __init__(self,
                 vocab: Vocab):
        self.sequence = SETTINGS.inference_engine.inference_sequence
        self.sr = SETTINGS.audio.sample_rate
        self.vocab = vocab


class WordStitcher(Stitcher):
    def __init__(self,
                 word_searcher: WordTranscriptSearcher,
                 **kwargs):
        super().__init__(**kwargs)
        self.word_searcher = word_searcher

    def stitch(self, dataset: AudioDataset):
        dir_path = Path("test/test_data/generated_data")
        dir_path.mkdir(exist_ok=True)

        sample_set = [[] for _ in range(len(self.vocab))]

        for sample in dataset:
            for label, start_timestamp, end_timestamp in sample.label_data.timestamp_list:
                print(label, self.vocab[label])
                start_idx = int(start_timestamp * self.sr / 1000)
                print(f"\tstart: {start_timestamp} ({start_idx})")
                end_idx = int(end_timestamp * self.sr / 1000)
                print(f"\tend: {end_timestamp} ({end_idx})")
                print(f"\taudio_length: {1000*len(sample.audio_data[start_idx:end_idx])/self.sr} ms")

                output_path = dir_path / \
                    f"{self.vocab[label]}_{sample.metadata.audio_id}_{start_idx}_{end_idx}.wav"
                soundfile.write(output_path, sample.audio_data[start_idx:end_idx].numpy(), self.sr)

                print(f"\tfile generated at {output_path}")

                sample_set[label].append(sample.audio_data[start_idx:end_idx])

        # TODO:: make sure enough samples there are for each
        sample_list = []
        for element in self.sequence:
            sample_list.append(sample_set[element])

        for sample_idx, sample_set in enumerate(itertools.product(*sample_list)):
            output_path = dir_path / f"{sample_idx}.wav"
            soundfile.write(output_path, torch.cat(sample_set).numpy(), self.sr)

            print(f"wake wordsample generated at {output_path}")
