from pathlib import Path
from typing import List

import torch

import howl
from howl.data.common.example import AudioClipExample
from howl.data.common.labeler import PhoneticFrameLabeler
from howl.data.common.metadata import AudioClipMetadata
from howl.data.common.phone import PhonePhrase, PronunciationDictionary
from howl.data.dataset.dataset import AudioDataset
from howl.utils.audio_utils import silent_load

WAKEWORD = "the"
VOCAB = [WAKEWORD]


def test_audio_file_path():
    """Path of a single audio file"""
    return common_voice_dataset_path() / "clips/common_voice_en_20005954.mp3"


def test_data_path():
    """Test data folder path"""
    return howl.root_path() / "test/test_data"


def common_voice_dataset_path():
    """Test common voice dataset folder path"""
    return test_data_path() / "datasets/common-voice"


def howl_audio_datasets_path():
    """Test raw audio dataset folder path"""
    return test_data_path() / "datasets/howl_audio_datasets"


def get_num_of_lines(file_path: Path):
    """Get number of lines in the given file path"""
    with open(file_path) as file:
        lines = len(file.readlines())
    return lines


def compare_files(file_path_1: Path, file_path_2: Path):
    """Return True if the two files have same contents"""
    with open(file_path_1) as file_1:
        with open(file_path_2) as file_2:
            if file_1.readline() != file_2.readline():
                return False
    return True


def pronounce_dict():
    """Test pronounce dictionary"""
    phone_dict_file = test_data_path() / "pronounciation_dictionary.txt"
    return PronunciationDictionary.from_file(phone_dict_file)


def frame_labeller(vocab: List[str]):
    """Test FrameLabeller loaded for the given vocab"""
    test_pronounce_dict = pronounce_dict()
    adjusted_vocab = []
    for word in vocab:
        phone_phrase = test_pronounce_dict.encode(word)[0]
        adjusted_vocab.extend(list(str(phone) for phone in phone_phrase.phones))

    phone_phrases = [PhonePhrase.from_string(x) for x in adjusted_vocab]
    return PhoneticFrameLabeler(phone_phrases, test_pronounce_dict)


class TestDataset(AudioDataset[AudioClipMetadata]):
    """Sample dataset for testing"""

    __test__ = False

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        test_common_voice_dataset_path = common_voice_dataset_path()
        metadata1 = AudioClipMetadata(
            transcription="The applicants are invited for coffee and visa is given immediately.",
            path=test_common_voice_dataset_path / "clips/common_voice_en_20005954.mp3",
        )
        audio_data = torch.from_numpy(silent_load(str(metadata1.path), self.sample_rate, self.mono))
        sample1 = AudioClipExample(metadata=metadata1, audio_data=audio_data, sample_rate=self.sample_rate)

        metadata2 = AudioClipMetadata(
            transcription="The anticipated synergies of the two modes of transportation were entirely absent.",
            path=test_common_voice_dataset_path / "clips/common_voice_en_20009653.mp3",
        )
        audio_data = torch.from_numpy(silent_load(str(metadata2.path), self.sample_rate, self.mono))
        sample2 = AudioClipExample(metadata=metadata2, audio_data=audio_data, sample_rate=self.sample_rate)

        metadata3 = AudioClipMetadata(
            transcription="The fossil fuels include coal, petroleum and natural gas.",
            path=test_common_voice_dataset_path / "clips/common_voice_en_20009655.mp3",
        )
        audio_data = torch.from_numpy(silent_load(str(metadata3.path), self.sample_rate, self.mono))
        sample3 = AudioClipExample(metadata=metadata3, audio_data=audio_data, sample_rate=self.sample_rate)

        self.samples = [sample1, sample2, sample3]
        self.metadata_list = [metadata1, metadata2, metadata3]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx) -> AudioClipExample:
        return self.samples[idx]


class TestNet(torch.nn.Module):
    """Two layer DNN for test case"""

    def __init__(self, input_dim=3, hidden_dim=3, output_dim=3):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super().__init__()
        self.linear1 = torch.nn.Linear(input_dim, hidden_dim)
        self.linear2 = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, input_tensor):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        h_relu = self.linear1(input_tensor).clamp(min=0)
        y_pred = self.linear2(h_relu)
        return y_pred
