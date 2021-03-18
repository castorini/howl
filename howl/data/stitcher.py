
import itertools
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import soundfile
import torch
from howl.data.dataset import (AudioClipDataset, AudioClipExample,
                               AudioClipMetadata, AudioDataset,
                               WakeWordDataset)
from howl.data.searcher import WordTranscriptSearcher
from howl.data.tokenize import Vocab
from howl.settings import SETTINGS

__all__ = ['WordStitcher']


@dataclass
class FrameLabelledSample:
    audio_data: torch.Tensor
    audio_length_ms: float
    end_timestamps: Optional[List[float]]
    label: int


class Stitcher:
    def __init__(self,
                 vocab: Vocab):
        self.sequence = SETTINGS.inference_engine.inference_sequence
        self.sr = SETTINGS.audio.sample_rate
        self.vocab = vocab
        self.wakeword = ' '.join(self.vocab[x]
                                 for x in self.sequence)


class WordStitcher(Stitcher):
    def __init__(self,
                 word_searcher: WordTranscriptSearcher,
                 **kwargs):
        super().__init__(**kwargs)
        self.word_searcher = word_searcher

    def concatenate_end_timestamps(self, end_timestamps_list: List[List[float]]):
        concatnated_timestamps = []
        last_timestamp = 0
        for end_timestamps in end_timestamps_list:
            for timestamp in end_timestamps:
                concatnated_timestamps.append(timestamp + last_timestamp)

            # when stitching space will be added between the vocabs
            # therefore last timestamp is repeated once to make up for the added space
            concatnated_timestamps.append(concatnated_timestamps[-1])
            last_timestamp = concatnated_timestamps[-1]

        return concatnated_timestamps[:-1]  # discard last space timestamp

    def stitch(self, * datasets: AudioDataset):
        dir_path = Path("test/test_data/generated_data")
        dir_path.mkdir(exist_ok=True)

        sample_set = [[] for _ in range(len(self.vocab))]

        for dataset in datasets:
            for sample in dataset:
                for idx, (label, char_indices) in enumerate(sample.label_data.char_indices):

                    vocab_start_idx = char_indices[0] - 1 if char_indices[0] > 0 else 0
                    start_timestamp = sample.metadata.end_timestamps[vocab_start_idx]
                    end_timestamp = sample.metadata.end_timestamps[char_indices[-1]]

                    print(label, self.vocab[label])
                    audio_start_idx = int(start_timestamp * self.sr / 1000)
                    # print(f"\tstart: {start_timestamp} ({audio_start_idx})")
                    audio_end_idx = int(end_timestamp * self.sr / 1000)
                    # print(f"\tend: {end_timestamp} ({audio_end_idx})")
                    # print(f"\taudio_length: {1000*len(sample.audio_data[audio_start_idx:audio_end_idx])/self.sr} ms")

                    output_path = dir_path / \
                        f"{self.vocab[label]}_{sample.metadata.audio_id}_{audio_start_idx}_{audio_end_idx}.wav"
                    soundfile.write(output_path, sample.audio_data[audio_start_idx:audio_end_idx].numpy(), self.sr)

                    print(f"\tfile generated at {output_path}")

                    adjusted_end_timestamps = []
                    for char_idx in char_indices:
                        adjusted_end_timestamps.append(sample.metadata.end_timestamps[char_idx] - start_timestamp)

                    sample_set[label].append(FrameLabelledSample(
                        sample.audio_data[audio_start_idx:audio_end_idx], end_timestamp-start_timestamp, adjusted_end_timestamps, label))

        # TODO:: make sure enough samples there are for each
        sample_list = [sample_set[element] for element in self.sequence]

        self.stitched_samples = []
        for sample_idx, sample_set in enumerate(itertools.product(*sample_list)):

            metatdata = AudioClipMetadata(
                path=dir_path / f"{sample_idx}.wav",
                transcription=self.wakeword,
                end_timestamps=self.concatenate_end_timestamps(
                    [labelled_data.end_timestamps for labelled_data in sample_set])
            )

            stitched_sample = AudioClipExample(
                metadata=metatdata,
                audio_data=torch.cat([labelled_data.audio_data for labelled_data in sample_set]),
                sample_rate=self.sr)

            self.stitched_samples.append(stitched_sample)

        for sample in self.stitched_samples:
            print(f"wake wordsample generated at {stitched_sample.metadata.path}")
            soundfile.write(stitched_sample.metadata.path, stitched_sample.audio_data.numpy(), self.sr)
