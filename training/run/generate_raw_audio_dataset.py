import os
from pathlib import Path

import howl
from howl.dataset.raw_audio_dataset_generator import RawAudioDatasetGenerator, SampleType
from howl.dataset_loader.dataset_loader_factory import DatasetLoaderType
from howl.settings import SETTINGS
from howl.utils import logging_utils
from howl.utils.args_utils import ArgOption, ArgumentParserBuilder


def main(
    input_audio_dataset_path: Path,
    dataset_loader_type: DatasetLoaderType,
    datasets_dir_path: Path,
    positive_pct: int,
    negative_pct: int,
):
    """Generate audio dataset for howl from the given audio dataset
    one dataset will be generated for positive samples and another dataset will be generated for negative samples
    they can be found under datasets_dir_path as <wakeword>-positive and <wakeword>-negative

    Args:
        input_audio_dataset_path: original audio dataset of which howl dataset will be created from
        dataset_loader_type: type of the dataset loader to use
        datasets_dir_path: path of the dir which the generated howl datasets are stored
        positive_pct: the percentage of the dataset to process for positive samples
        negative_pct: the percentage of the dataset to process for negative samples
    """
    logger = logging_utils.setup_logger(os.path.basename(__file__))

    raw_dataset_generator = RawAudioDatasetGenerator(input_audio_dataset_path, dataset_loader_type, logger)
    datasets_dir_path.mkdir(exist_ok=True)

    wakeword = ""
    for idx, sequence in enumerate(SETTINGS.inference_engine.inference_sequence):
        wakeword += SETTINGS.training.vocab[sequence]
        if idx != len(SETTINGS.inference_engine.inference_sequence) - 1:
            wakeword += "_"

    logger.info(f"Generating raw audio dataset for {wakeword} from {input_audio_dataset_path}")
    wakeword_dataset_path = datasets_dir_path / wakeword
    wakeword_dataset_path.mkdir()

    if positive_pct > 0:
        positive_dataset_path = wakeword_dataset_path / "positive"
        logger.info(f"Generating positive dataset: {positive_dataset_path}")
        positive_dataset_path.mkdir()
        raw_dataset_generator.generate_datasets(positive_dataset_path, SampleType.POSITIVE, percentage=positive_pct)
    else:
        logger.warning("Skipping positive dataset generation because positive_pct is 0")

    if negative_pct > 0:
        negative_dataset_path = wakeword_dataset_path / "negative"
        logger.info(f"Generating negative dataset: {negative_dataset_path}")
        negative_dataset_path.mkdir()
        # skip statistics computation for the negative dataset
        # because the process is slow due to its size and it doesn't have any vocab
        raw_dataset_generator.generate_datasets(
            negative_dataset_path, SampleType.NEGATIVE, percentage=negative_pct, print_statistics=False
        )
    else:
        logger.warning("Skipping negative dataset generation because positive_pct is 0")


def setup():
    """Parse the arguments"""

    input_audio_dataset_path = "/data/common-voice"
    dataset_loader_type = DatasetLoaderType.COMMON_VOICE_DATASET_LOADER.value
    datasets_dir_path = str(howl.datasets_path())
    positive_pct = 100
    negative_pct = 100

    apb = ArgumentParserBuilder()
    apb.add_options(
        ArgOption(
            "--input-audio-dataset-path",
            "-i",
            type=str,
            default=input_audio_dataset_path,
            help="location of the input audio dataset (default: /data/common-voice)",
        ),
        ArgOption(
            "--dataset-loader-type",
            type=str,
            default=dataset_loader_type,
            choices=[e.value for e in DatasetLoaderType],
            help="type of dataset loader to use",
        ),
        ArgOption(
            "--datasets-dir-path",
            "-o",
            type=str,
            default=datasets_dir_path,
            help="path of the dir which the generated howl datasets are stored (default: dataset)",
        ),
        ArgOption(
            "--positive-pct",
            type=int,
            default=positive_pct,
            help="The percentage of the dataset to process for positive samples [0, 100]",
        ),
        ArgOption(
            "--negative-pct",
            type=int,
            default=negative_pct,
            help="The percentage of the dataset to process for negative samples [0, 100]",
        ),
    )
    raw_args = apb.parser.parse_args()

    if raw_args.positive_pct < 0 or raw_args.positive_pct > 100:
        raise ValueError("Argument positive-pct must be in percentage; [0, 100]")

    if raw_args.negative_pct < 0 or raw_args.negative_pct > 100:
        raise ValueError("Argument negative-pct must be in percentage; [0, 100]")

    return raw_args


if __name__ == "__main__":
    args = setup()

    main(
        Path(args.input_audio_dataset_path),
        DatasetLoaderType(args.dataset_loader_type),
        Path(args.datasets_dir_path),
        args.positive_pct,
        args.negative_pct,
    )
