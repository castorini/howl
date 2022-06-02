import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import soundfile
import torch
from tqdm import tqdm

from howl.data.common.example import AudioClipExample
from howl.data.common.metadata import AudioClipMetadata
from howl.data.common.vocab import Vocab
from howl.data.dataset.dataset import AudioDataset
from howl.settings import SETTINGS
from howl.utils.sphinx_keyword_detector import SphinxKeywordDetector

__all__ = ["WordStitcher"]


@dataclass
class FrameLabelledSample:
    """Frame level information"""

    audio_data: torch.Tensor
    audio_length_ms: float
    end_timestamps: Optional[List[float]]
    label: int


class Stitcher:
    """Stitches audio clips to generate custom audio clip"""

    def __init__(self, vocab: Vocab, inference_sequence: List[int] = None, validate_stitched_sample: bool = True):
        """Base Stitcher class

        Args:
            vocab (Vocab): vocab containing wakeword
            inference_sequence (List[int]): sequence of vocab that makes up the wakeword
            validate_stitched_sample (bool, optional): drop invalid stitched samples
                                                       through secondary keyword detection step
        """
        self.inference_sequence = inference_sequence
        if self.inference_sequence is None:
            self.inference_sequence = SETTINGS.inference_engine.inference_sequence
        self.sample_rate = SETTINGS.audio.sample_rate
        self.vocab = vocab
        self.wakeword = " ".join(self.vocab[x] for x in self.inference_sequence)

        self.validate_stitched_sample = validate_stitched_sample
        self.keyword_detector = []
        if self.validate_stitched_sample:
            for word in self.inference_sequence:
                self.keyword_detector.append(SphinxKeywordDetector(self.vocab[word]))


class WordStitcher(Stitcher):
    """Stitches word-level audio clips to generate custom audio clip"""

    # TODO: WordStitcher needs to be refactored
    # pylint: disable=too-many-branches
    # pylint: disable=too-many-statements

    def __init__(self, **kwargs):
        """Initialize WordStitcher"""
        super().__init__(**kwargs)
        self.stitched_samples = []

    def concatenate_end_timestamps(self, end_timestamps_list: List[List[float]]) -> List[float]:
        """concatenate given list of end timestamps for single audio sample

        Args:
            end_timestamps_list (List[List[float]]): list of timestamps to concatenate

        Returns:
            List[float]: concatenated end timestamps
        """
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

    def generate_stitched_audio_samples(
        self, num_stitched_samples: int, stitched_audio_dir: Path, *datasets: AudioDataset
    ):
        """collect vocab samples from datasets and generate stitched wakeword samples

        Args:
            num_stitched_samples (int): number of stitched wakeword samples to generate
            stitched_audio_dir (Path): folder which the stitched audio samples will be saved
            datasets (Path): list of datasets to collect vocab samples from
        """
        sample_set = [[] for _ in range(len(self.vocab))]

        for dataset in datasets:
            # for each audio sample, collect vocab audio sample based on alignment
            for sample in dataset:
                for (label, char_indices) in sample.label.char_indices:
                    vocab_start_idx = char_indices[0] - 1 if char_indices[0] > 0 else 0
                    start_timestamp = sample.metadata.end_timestamps[vocab_start_idx]
                    end_timestamp = sample.metadata.end_timestamps[char_indices[-1]]

                    audio_start_idx = int(start_timestamp * self.sample_rate / 1000)
                    audio_end_idx = int(end_timestamp * self.sample_rate / 1000)

                    adjusted_end_timestamps = []
                    for char_idx in char_indices:
                        adjusted_end_timestamps.append(sample.metadata.end_timestamps[char_idx] - start_timestamp)

                    sample_set[label].append(
                        FrameLabelledSample(
                            sample.audio_data[audio_start_idx:audio_end_idx],
                            end_timestamp - start_timestamp,
                            adjusted_end_timestamps,
                            label,
                        )
                    )

        # reorganize and make sure there are enough samples for each vocab
        sample_lists = []
        for element in self.inference_sequence:
            print(f"number of samples for vocab {self.vocab[element]}: {len(sample_set[element])}")
            assert len(sample_set[element]) > 0, "There must be at least one sample for each vocab"
            sample_lists.append(sample_set[element])

        # generate AudioClipExample for each vocab sample
        self.stitched_samples = []

        pbar = tqdm(total=num_stitched_samples, desc="Generating stitched samples")
        sample_idx = 0
        num_skipped_samples = 0
        while True:
            if sample_idx == num_stitched_samples:
                break

            sample_set = []
            for sample_list in sample_lists:
                sample_set.append(random.choice(sample_list))

            audio_data = torch.cat([labelled_data.audio_data for labelled_data in sample_set])

            if self.validate_stitched_sample:
                temp_audio_file_path = "/tmp/temp.wav"
                soundfile.write(temp_audio_file_path, audio_data.numpy(), self.sample_rate)

                keyword_exists = True
                for detector in self.keyword_detector:
                    # sphinx keyword detection may not be sufficient for audio with repeated words
                    if len(detector.detect(temp_audio_file_path)) == 0:
                        keyword_exists = False
                        break

                if keyword_exists:
                    num_skipped_samples += 1
                    continue

            metatdata = AudioClipMetadata(
                path=Path(stitched_audio_dir / f"{sample_idx}").with_suffix(".wav"),
                transcription=self.wakeword,
                end_timestamps=self.concatenate_end_timestamps(
                    [labelled_data.end_timestamps for labelled_data in sample_set]
                ),
            )

            # TODO:: dataset writer load the samples upon write and does not use data in memory
            #        writer classes need to be refactored to use audio data if exist
            soundfile.write(metatdata.path, audio_data.numpy(), self.sample_rate)

            stitched_sample = AudioClipExample(metadata=metatdata, audio_data=audio_data, sample_rate=self.sample_rate)

            self.stitched_samples.append(stitched_sample)

            sample_idx += 1
            pbar.update()

        if self.validate_stitched_sample:
            print(
                f"While generating {num_stitched_samples} stitched samples, "
                f"{num_skipped_samples} are filtered by keyword detection"
            )
