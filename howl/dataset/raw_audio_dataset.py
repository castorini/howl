from functools import lru_cache

import torch

from howl.data.common.example import AudioClipExample
from howl.data.dataset.dataset import AudioClipMetadata, AudioDataset
from howl.settings import SETTINGS
from howl.utils.audio import silent_load


class RawAudioDataset(AudioDataset[AudioClipMetadata]):
    """raw audio dataset (dataset without transcription alignment)"""

    METADATA_FILE_NAME_TEMPLATE = "metadata-{dataset_split}.jsonl"

    @lru_cache(maxsize=SETTINGS.cache.cache_size)
    def __getitem__(self, idx) -> AudioClipExample:
        """Get sample for the given idx"""
        metadata = self.metadata_list[idx]
        audio_data = silent_load(str(metadata.path), self.sample_rate, self.mono)
        return AudioClipExample(
            metadata=metadata, audio_data=torch.from_numpy(audio_data), sample_rate=self.sample_rate,
        )
