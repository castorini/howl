from enum import Enum, unique
from pathlib import Path

from howl.dataset_loader.aligned_audio_dataset_loader import AlignedAudioDatasetLoader
from howl.dataset_loader.common_voice_dataset_loader import CommonVoiceDatasetLoader
from howl.dataset_loader.dataset_loader import AudioDatasetLoader
from howl.dataset_loader.raw_audio_dataset_loader import RawAudioDatasetLoader


@unique
class DatasetLoaderType(str, Enum):
    """String based Enum of different dataset loader type"""

    COMMON_VOICE_DATASET_LOADER = CommonVoiceDatasetLoader.NAME
    RAW_AUDIO_DATASET_LOADER = RawAudioDatasetLoader.NAME
    ALIGNED_AUDIO_DATASET_LOADER = AlignedAudioDatasetLoader.NAME


def get_dataset_loader(dataset_loader_type: DatasetLoaderType, dataset_path: Path) -> AudioDatasetLoader:
    """Get dataset loader of the given type

    Args:
        dataset_loader_type: type of the dataset loader
        dataset_path: location of the dataset

    Returns:
        Dataset loader of the given type
    """
    if dataset_loader_type == DatasetLoaderType.COMMON_VOICE_DATASET_LOADER:
        dataset_loader = CommonVoiceDatasetLoader(dataset_path)
    elif dataset_loader_type == DatasetLoaderType.RAW_AUDIO_DATASET_LOADER:
        dataset_loader = RawAudioDatasetLoader(dataset_path)
    else:
        raise RuntimeError(f"Given dataset loader type is invalid: {dataset_loader_type}")

    return dataset_loader
