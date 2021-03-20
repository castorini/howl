from typing import List

from pydantic import BaseSettings

__all__ = ['AudioSettings', 'RawDatasetSettings', 'DatasetSettings', 'SETTINGS']


class CacheSettings(BaseSettings):
    cache_size: int = 128144


class AudioSettings(BaseSettings):
    sample_rate: int = 16000
    use_mono: bool = True


class AudioTransformSettings(BaseSettings):
    num_fft: int = 512
    num_mels: int = 80
    sample_rate: int = 16000
    hop_length: int = 200
    use_meyda_spectrogram: bool = False


class InferenceEngineSettings(BaseSettings):
    inference_weights: List[float] = None
    inference_sequence: List[int] = [0]
    inference_window_ms: float = 2000  # look at last of these seconds
    smoothing_window_ms: float = 50  # prediction smoothed
    tolerance_window_ms: float = 500  # negative label between words
    inference_threshold: float = 0  # positive label probability must rise above this threshold


class TrainingSettings(BaseSettings):
    seed: int = 0
    # TODO:: vocab should not belong to training
    vocab: List[str] = ['fire']
    num_epochs: int = 10
    num_labels: int = 2
    learning_rate: float = 1e-3
    device: str = 'cuda:0'
    batch_size: int = 16
    lr_decay: float = 0.75
    max_window_size_seconds: float = 0.75
    eval_window_size_seconds: float = 0.75
    eval_stride_size_seconds: float = 0.063
    weight_decay: float = 0
    convert_static: bool = False
    objective: str = 'frame'  # frame or ctc
    token_type: str = 'word'
    phone_dictionary: str = None
    use_noise_dataset: bool = False


class RawDatasetSettings(BaseSettings):
    common_voice_dataset_path: str = None
    wake_word_dataset_path: str = None
    keyword_voice_dataset_path: str = None
    noise_dataset_path: str = None


class DatasetSettings(BaseSettings):
    dataset_path: str = None


class HowlSettings:
    """Lazy-loaded class containing all required settings"""
    _audio: AudioSettings = None
    _audio_transform: AudioTransformSettings = None
    _inference_engine: InferenceEngineSettings = None
    _raw_dataset: RawDatasetSettings = None
    _dataset: DatasetSettings = None
    _cache: CacheSettings = None
    _training: TrainingSettings = None

    @property
    def audio(self) -> AudioSettings:
        if self._audio is None:
            self._audio = AudioSettings()
        return self._audio

    @property
    def audio_transform(self) -> AudioTransformSettings:
        if self._audio_transform is None:
            self._audio_transform = AudioTransformSettings()
        return self._audio_transform

    @property
    def inference_engine(self) -> InferenceEngineSettings:
        if self._inference_engine is None:
            self._inference_engine = InferenceEngineSettings()
        return self._inference_engine

    @property
    def raw_dataset(self) -> RawDatasetSettings:
        if self._raw_dataset is None:
            self._raw_dataset = RawDatasetSettings()
        return self._raw_dataset

    @property
    def dataset(self) -> DatasetSettings:
        if self._dataset is None:
            self._dataset = DatasetSettings()
        return self._dataset

    @property
    def cache(self) -> CacheSettings:
        if self._cache is None:
            self._cache = CacheSettings()
        return self._cache

    @property
    def training(self) -> TrainingSettings:
        if self._training is None:
            self._training = TrainingSettings()
        return self._training


KEY_TO_SETTINGS_CLASS = {
    '_audio': AudioSettings,
    '_audio_transform': AudioTransformSettings,
    '_inference_engine': InferenceEngineSettings,
    '_raw_dataset': RawDatasetSettings,
    '_dataset': DatasetSettings,
    '_cache': CacheSettings,
    '_training': TrainingSettings,
}

SETTINGS = HowlSettings()
