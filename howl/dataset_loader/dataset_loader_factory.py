from pathlib import Path

from howl.dataset.audio_dataset_constants import AudioDatasetType
from howl.dataset_loader.common_voice_dataset_loader import CommonVoiceDatasetLoader
from howl.dataset_loader.dataset_loader import AudioDatasetLoader
from howl.dataset_loader.howl_audio_dataset_loader import HowlAudioDatasetLoader


def get_dataset_loader(dataset_type: AudioDatasetType, dataset_path: Path) -> AudioDatasetLoader:
    """Get dataset loader of the given type

    Args:
        dataset_type: type of the dataset
        dataset_path: location of the dataset

    Returns:
        Dataset loader of the given type
    """
    if dataset_type == AudioDatasetType.COMMON_VOICE:
        dataset_loader = CommonVoiceDatasetLoader(dataset_path)
    elif dataset_type == AudioDatasetType.RAW:
        dataset_loader = HowlAudioDatasetLoader(AudioDatasetType.RAW, dataset_path)
    elif dataset_type == AudioDatasetType.ALIGNED:
        dataset_loader = HowlAudioDatasetLoader(AudioDatasetType.ALIGNED, dataset_path)
    else:
        raise RuntimeError(f"Given dataset type is invalid: {dataset_type}")

    return dataset_loader
