from enum import Enum, unique


@unique
class SampleType(str, Enum):
    """String based Enum for positive/negative sample types"""

    POSITIVE = "positive"
    NEGATIVE = "negative"


@unique
class AudioDatasetType(str, Enum):
    """String based Enum for audio dataset types"""

    COMMON_VOICE = "common-voice"
    RAW = "raw"
    ALIGNED = "aligned"


METADATA_FILE_NAME_TEMPLATES = {
    AudioDatasetType.RAW: "metadata-{dataset_split}.jsonl",
    AudioDatasetType.ALIGNED: "aligned-metadata-{dataset_split}.jsonl",
}
