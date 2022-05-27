from enum import Enum, unique
from functools import lru_cache

import torch

from howl.data.common.labeler import FrameLabeler
from howl.data.common.sample import Sample
from howl.data.dataset.dataset import AudioClipMetadata, AudioDataset
from howl.settings import SETTINGS
from howl.utils.audio_utils import silent_load


@unique
class AudioDatasetType(str, Enum):
    """String based Enum for audio dataset types"""

    RAW = "raw"
    ALIGNED = "aligned"


class AlignedAudioDataset(AudioDataset[AudioClipMetadata]):
    """audio dataset"""

    # TODO: AlignedAudioDataset class and AudioDataset should be combined into a single class

    METADATA_FILE_NAME_TEMPLATES = {
        AudioDatasetType.RAW: "metadata-{dataset_split}.jsonl",
        AudioDatasetType.ALIGNED: "aligned-metadata-{dataset_split}.jsonl",
    }

    def __init__(self, *args, labeler: FrameLabeler = None, **kwargs):
        """Initialize AudioDataset

        Args:
            labeler (FrameLabeler): if specified, sample instance will have frame-level
                                    labels constructed using the provided labeler
        """
        super().__init__(*args, **kwargs)
        self.labeler = labeler

    @classmethod
    def load_sample(
        cls, metadata: AudioClipMetadata, sample_rate: int, mono: bool, labeler: FrameLabeler = None
    ) -> Sample:
        """Load audio data which metadata points to as a Sample

        Args:
            metadata (AudioClipMetadata): metadata
            sample_rate (int): sample rate which the audio will be loaded with
            mono (bool): if True, load the audio data as single-channel (mono) audio
            labeler (FrameLabeler): if specified, sample instance will have frame-level
                                    labels constructed using the provided labeler

        Returns:
            Sample instance created from the given metadata, sample rate and the loaded audio data
        """
        audio_data = silent_load(str(metadata.path), sample_rate, mono)
        label = None
        if labeler is not None:
            label = labeler.compute_frame_labels(metadata)

        return Sample(metadata=metadata, audio_data=torch.from_numpy(audio_data), sample_rate=sample_rate, label=label)

    @lru_cache(maxsize=SETTINGS.cache.cache_size)
    def __getitem__(self, idx) -> Sample:
        """Get sample for the given idx"""
        return AlignedAudioDataset.load_sample(
            self.metadata_list[idx], sample_rate=self.sample_rate, mono=self.mono, labeler=self.labeler
        )
