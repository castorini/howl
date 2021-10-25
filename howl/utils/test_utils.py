import howl


def test_data_path():
    return howl.root_path() / "test/test_data"


def common_voice_dataset_path():
    return test_data_path() / "dataset/common-voice"
