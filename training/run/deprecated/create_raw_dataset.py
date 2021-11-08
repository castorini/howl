import logging
import os
from pathlib import Path

from howl.context import InferenceContext
from howl.data.common.metadata import AudioClipMetadata
from howl.data.dataset.dataset import AudioDataset
from howl.data.dataset.dataset_writer import AudioDatasetWriter
from howl.dataset_loader.dataset_loader_factory import DatasetLoaderType, get_dataset_loader
from howl.settings import SETTINGS
from howl.utils import hash_utils, logging_utils
from training.run.args import ArgumentParserBuilder, opt


# TODO: to be replaced to dataset.print_stats
def print_stats(header: str, context: InferenceContext, *datasets: AudioDataset, compute_length=True):
    """Print statistics for the give datasets

    Args:
        header: additional text message to prepend
        context: inference context
        *datasets: datasets of which statistics will be printed
        compute_length: log total length of the audio
    """
    word_searcher = context.searcher if context.token_type == "word" else None
    for dataset in datasets:
        logging.info(
            f"{header} ({dataset.set_type}) "
            f"statistics: {dataset.compute_statistics(word_searcher=word_searcher, compute_length=compute_length)}"
        )


def main():
    """
    This scripts processes given audio dataset and creates datasets that howl can take in
    distributions can be customized using negative-pct and positive-pct arguments

    sample command:
    VOCAB='["fire"]' INFERENCE_SEQUENCE=[0] DATASET_PATH=data/fire-positive \
    python -m training.run.create_raw_dataset -i ~/path/to/common-voice --dataset-loader-type mozilla-cv \
    --positive-pct 100 --negative-pct 0
    """

    def filter_fn(metadata: AudioClipMetadata):
        """Distribute given audio data into different bucket based on the configuration and
        return True if it needs to be included

        Args:
            metadata: metadata of the audio sample

        Returns:
            True if it needs to be included for the dataset
        """
        bucket = hash_utils.sha256_int(metadata.path.stem) % 100
        if bucket < args.negative_pct:
            # drop the samples whose transcript is wakeword
            return not ctx.searcher.search(metadata.transcription.lower())
        if bucket < args.positive_pct:
            # select all sample whose transcript contains at least one of the vocabs
            return ctx.searcher.contains_any(metadata.transcription.lower())
        return False

    apb = ArgumentParserBuilder()
    apb.add_options(
        opt(
            "--negative-pct", type=int, default=1, help="The percentage of the dataset to check for negative examples.",
        ),
        opt(
            "--positive-pct",
            type=int,
            default=100,
            help="The percentage of the dataset to check for positive examples.",
        ),
        opt("--input-audio-dataset-path", "-i", type=str, help="location of the input audio dataset",),
        opt(
            "--dataset-loader-type",
            type=str,
            default=DatasetLoaderType.COMMON_VOICE_DATASET_LOADER.value,
            choices=[e.value for e in DatasetLoaderType],
            help="type of dataset loader to use",
        ),
    )
    args = apb.parser.parse_args()
    if args.input_audio_dataset_path is None:
        args.input_audio_dataset_path = SETTINGS.raw_dataset.common_voice_dataset_path

    logger = logging_utils.setup_logger(os.path.basename(__file__))

    dataset_loader_type = DatasetLoaderType(args.dataset_loader_type)
    dataset_loader = get_dataset_loader(dataset_loader_type, Path(args.input_audio_dataset_path), logger)
    ds_kwargs = dict(sample_rate=SETTINGS.audio.sample_rate, mono=SETTINGS.audio.use_mono)
    train_ds, dev_ds, test_ds = dataset_loader.load_splits(**ds_kwargs)

    ctx = InferenceContext(SETTINGS.training.vocab, token_type=SETTINGS.training.token_type)
    if dataset_loader_type == DatasetLoaderType.COMMON_VOICE_DATASET_LOADER:
        train_ds = train_ds.filter(filter_fn)
        dev_ds = dev_ds.filter(filter_fn)
        test_ds = test_ds.filter(filter_fn)

    word_searcher = ctx.searcher if ctx.token_type == "word" else None
    for dataset in train_ds, dev_ds, test_ds:
        dataset.print_stats(logger, word_searcher=word_searcher, compute_length=True)
        logger.info(f"Generating {dataset.split.value} dataset")
        AudioDatasetWriter(dataset).write(Path(SETTINGS.dataset.dataset_path))


if __name__ == "__main__":
    main()
