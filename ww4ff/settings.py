from pathlib import Path

from pydantic import BaseSettings


__all__ = ['AudioSettings', 'RawDatasetSettings', 'DatasetSettings', 'SETTINGS']


class AudioSettings(BaseSettings):
    sample_rate: int = 16000
    use_mono: bool = True


class RawDatasetSettings(BaseSettings):
    common_voice_dataset_path: Path
    wake_word_dataset_path: Path


class DatasetSettings(BaseSettings):
    dataset_path: Path


class LazySettingsSingleton:
    _audio: AudioSettings = None
    _raw_dataset: RawDatasetSettings = None
    _dataset: DatasetSettings = None

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


SETTINGS = LazySettingsSingleton()
