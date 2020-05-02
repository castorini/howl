from pathlib import Path

from pydantic import BaseSettings


__all__ = ['AudioSettings', 'RawDatasetSettings', 'DatasetSettings', 'SETTINGS']


class CacheSettings(BaseSettings):
    cache_size: int = 16384


class AudioSettings(BaseSettings):
    sample_rate: int = 16000
    use_mono: bool = True


class TrainingSettings(BaseSettings):
    seed: int = 0
    wake_word: str = 'Hey Firefox'
    num_epochs: int = 10
    learning_rate: float = 1e-3
    device: str = 'cuda:0'
    batch_size: int = 16
    max_window_size_seconds: float = 2.5
    eval_window_size_seconds: float = 2.5
    eval_stride_size_seconds: float = 0.1
    weight_decay: float = 0


class RawDatasetSettings(BaseSettings):
    common_voice_dataset_path: Path
    wake_word_dataset_path: Path


class DatasetSettings(BaseSettings):
    dataset_path: Path


class LazySettingsSingleton:
    _audio: AudioSettings = None
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


SETTINGS = LazySettingsSingleton()
