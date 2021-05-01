import json
import logging
from multiprocessing import Pool
from pathlib import Path
from typing import Tuple

from tqdm import tqdm

from howl.data.common.metadata import AudioClipMetadata
from howl.data.dataset.dataset import AudioClipDataset, DatasetType
from howl.data.dataset.dataset_loader import RegisteredPathDatasetLoader
from howl.utils.transcribe import SpeechToText

__all__ = [
    "HeySnipsWakeWordLoader",
]


def transcribe_hey_snips_audio(path, metadata):
    stt = SpeechToText()
    path = (path / metadata["audio_file_path"]).absolute()
    transcription = "hey snips"
    if metadata["is_hotword"] == 0:  # negative sample
        transcription = stt.transcribe(path)

    return path, transcription


class HeySnipsWakeWordLoader(RegisteredPathDatasetLoader, name="hey-snips"):
    def __init__(self, num_processes=8):
        self.stt = SpeechToText()
        self.num_processes = num_processes
        self.pool = Pool(processes=self.num_processes)

    def load_splits(self, path: Path, **dataset_kwargs) -> Tuple[AudioClipDataset, AudioClipDataset, AudioClipDataset]:
        self.path = path

        def load(filename, set_type):
            logging.info(f"Loading split {filename}...")

            metadata_list = []
            with open(str(self.path / filename)) as f:
                raw_metadata_list = json.load(f)

            total = len(raw_metadata_list)

            transcription_fail_count = 0
            pbar = tqdm(range(0, total, self.num_processes), desc=f"{filename}")
            for starting_idx in pbar:
                transcription_results = []
                for metadata in raw_metadata_list[starting_idx : starting_idx + self.num_processes]:
                    transcription_results.append(
                        self.pool.apply_async(transcribe_hey_snips_audio, (self.path, metadata,))
                    )

                processing_count = starting_idx + len(transcription_results)

                for result in transcription_results:
                    pbar.set_postfix(dict(transcription_fail_rate=f"{transcription_fail_count}/{processing_count}"))
                    path, transcription = result.get()

                    if transcription == "":
                        transcription_fail_count += 1
                        continue

                    metadata_list.append(AudioClipMetadata(path=path, transcription=transcription))

            logging.info(f"{transcription_fail_count}/{total} samples have empty transcription")

            return AudioClipDataset(metadata_list=metadata_list, set_type=set_type, **dataset_kwargs)

        assert path.exists(), "dataset path doesn't exist"
        filenames = ("train.json", "dev.json", "test.json")
        assert all((path / x).exists() for x in filenames), "dataset missing metadata"

        return (
            load("train.json", DatasetType.TRAINING),
            load("dev.json", DatasetType.DEV),
            load("test.json", DatasetType.TEST),
        )
