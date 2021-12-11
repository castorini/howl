from functools import lru_cache

import torch

from howl.data.common.example import AudioClipExample
from howl.data.dataset.dataset import AudioClipMetadata, AudioDataset
from howl.settings import SETTINGS
from howl.utils.audio import silent_load


class RawAudioDataset(AudioDataset[AudioClipMetadata]):
    """raw audio dataset (dataset without transcription alignment)"""

    METADATA_FILE_NAME_TEMPLATE = "metadata-{dataset_split}.jsonl"

    @classmethod
    def load_sample(cls, metadata: AudioClipMetadata, sample_rate: int, mono: bool):
        """Generate audio sample from the metadata"""
        audio_data = silent_load(str(metadata.path), sample_rate, mono)
        return AudioClipExample(metadata=metadata, audio_data=torch.from_numpy(audio_data), sample_rate=sample_rate,)

    @lru_cache(maxsize=SETTINGS.cache.cache_size)
    def __getitem__(self, idx) -> AudioClipExample:
        """Get sample for the given idx"""
        return RawAudioDataset.load_sample(self.metadata_list[idx], sample_rate=self.sample_rate, mono=self.mono)
