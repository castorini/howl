from typing import List

from pydantic import BaseModel


class CacheSettings(BaseModel):
    """Base settings for cache"""

    cache_size: int = 128144


class AudioSettings(BaseModel):
    """Base settings for loading audio file"""

    sample_rate: int = 16000
    use_mono: bool = True


class ContextSettings(BaseModel):
    """Base settings for the system"""

    seed: int = 0
    vocab: List[str] = [" hey", "fire", "fox"]
    sequence: List[int] = [0, 1, 2]
    # type of the token that the system will be based on
    # word: training and inference will be achieved at word level
    # phone: training and inference will be achieved at phoneme level
    token_type: str = "word"
    # phone dictionary file path
    phone_dictionary_path: str = None
    audio_settings: AudioSettings = AudioSettings()


class InferenceEngineSettings(BaseModel):
    """Base settings for inference engine"""

    # if True, FrameInferenceEngine will be used over the base InferenceEngine
    per_frame: bool = False
    # weighting on prediction (model output)
    inference_weights: List[float] = None
    # InferenceEngine says wake word is present
    # if a sequence of predictions from the last INFERENCE_WINDOW_MS audio data matches the target sequence
    inference_window_ms: float = 2000
    # predictions are smoothed over SMOOTHING_WINDOW_MS before the final labels are computed
    smoothing_window_ms: float = 50
    # negative labels are ignored as long as they don't last for TOLERANCE_WINDOW_MS
    tolerance_window_ms: float = 500
    # prediction probability for positive labels must be above this threshold
    inference_threshold: float = 0


class AudioTransformSettings(BaseModel):
    """Base settings for audio transform"""

    num_fft: int = 512
    num_mels: int = 40
    hop_length: int = 200
    use_meyda_spectrogram: bool = False


class DatasetSettings(BaseModel):
    """Base settings for dataset"""

    path: str = None
    audio_transform_settings: AudioTransformSettings = AudioTransformSettings()


class TrainingSettings(BaseModel):
    """Base settings for training"""

    batch_size: int = 16
    learning_rate: float = 1e-3
    num_epochs: int = 10
    lr_decay: float = 0.75
    weight_decay: float = 0
    use_noise_dataset: bool = False
    noise_datasets: List[DatasetSettings] = []
    train_datasets: List[DatasetSettings] = []
    val_datasets: List[DatasetSettings] = []
    test_datasets: List[DatasetSettings] = []
    inference_engine_settings: InferenceEngineSettings = InferenceEngineSettings()
    cache_settings: CacheSettings = CacheSettings()
    context_settings: ContextSettings = ContextSettings()


class InferenceSettings(BaseModel):
    """Base settings for inference"""

    inference_engine_settings: InferenceEngineSettings = InferenceEngineSettings()
    context_settings: ContextSettings = ContextSettings()
