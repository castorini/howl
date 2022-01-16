from functools import lru_cache

import torch

from howl.data.common.sample import Sample
from howl.data.dataset.dataset import AudioClipMetadata, AudioDataset
from howl.settings import SETTINGS
from howl.utils.audio_utils import silent_load


class RawAudioDataset(AudioDataset[AudioClipMetadata]):
    """raw audio dataset (dataset without transcription alignment)"""

    METADATA_FILE_NAME_TEMPLATE = "metadata-{dataset_split}.jsonl"

    @classmethod
    def load_sample(cls, metadata: AudioClipMetadata, sample_rate: int, mono: bool) -> Sample:
        """Load audio data which metadata points to as a Sample

        Args:
            metadata (AudioClipMetadata): metadta
            sample_rate (int): sample rate which the audio will be loaded with
            mono (bool): if True, load the audio data as single-channel (mono) audio

        Returns:
            Sample instance created from the given metadata, sample rate and the loaded audio data
        """
        audio_data = silent_load(str(metadata.path), sample_rate, mono)

        return Sample(metadata=metadata, audio_data=torch.from_numpy(audio_data), sample_rate=sample_rate)

    @lru_cache(maxsize=SETTINGS.cache.cache_size)
    def __getitem__(self, idx) -> Sample:
        """Get sample for the given idx"""
        return RawAudioDataset.load_sample(self.metadata_list[idx], sample_rate=self.sample_rate, mono=self.mono)
