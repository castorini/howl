from typing import List

from pydantic import BaseModel

# WIP; please use settings.py


class CacheConfig(BaseModel):
    """Base config for cache"""

    cache_size: int = 128144


class AudioConfig(BaseModel):
    """Base config for loading audio file"""

    sample_rate: int = 16000
    use_mono: bool = True


class ContextConfig(BaseModel):
    """Base config for the system"""

    seed: int = 0
    vocab: List[str] = None
    sequence: List[int] = None
    # type of the token that the system will be based on
    # word: training and inference will be achieved at word level
    # phone: training and inference will be achieved at phoneme level
    token_type: str = "word"
    # phone dictionary file path
    phone_dictionary_path: str = None


class InferenceEngineConfig(BaseModel):
    """Base config for inference engine"""

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


class AudioTransformConfig(BaseModel):
    """Base config for audio transform"""

    num_fft: int = 512
    num_mels: int = 40
    hop_length: int = 200
    use_meyda_spectrogram: bool = False


class DatasetConfig(BaseModel):
    """Base config for dataset"""

    path: str = None
    audio_config: AudioConfig = AudioConfig()
    audio_transform_config: AudioTransformConfig = AudioTransformConfig()


class ModelConfig(BaseModel):
    """Base config for model"""

    architecture: str = "res8"


class TrainingConfig(BaseModel):
    """Base config for training"""

    batch_size: int = 16
    learning_rate: float = 0.01
    num_epochs: int = 10
    lr_decay: float = 0.955
    weight_decay: float = 0.00001
    use_noise_dataset: bool = False
    noise_datasets: List[DatasetConfig] = []
    train_datasets: List[DatasetConfig] = []
    val_datasets: List[DatasetConfig] = []
    test_datasets: List[DatasetConfig] = []
    inference_engine_config: InferenceEngineConfig = InferenceEngineConfig()
    cache_config: CacheConfig = CacheConfig()
    model_config: ModelConfig = ModelConfig()
    context_config: ContextConfig = ContextConfig()
    workspace_path: str = None


class InferenceConfig(BaseModel):
    """Base config for inference"""

    inference_engine_config: InferenceEngineConfig = InferenceEngineConfig()
    context_config: ContextConfig = ContextConfig()
