import logging
from pathlib import Path

from howl.context import InferenceContext
from howl.data.common.metadata import AudioClipMetadata
from howl.data.dataset.dataset import AudioDataset
from howl.data.dataset.dataset_writer import AudioDatasetWriter
from howl.dataset.audio_dataset_constants import AudioDatasetType
from howl.dataset_loader.dataset_loader_factory import get_dataset_loader
from howl.settings import SETTINGS
from howl.utils import hash_utils
from howl.utils.args_utils import ArgOption, ArgumentParserBuilder
from howl.utils.logger import Logger


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
    python -m training.run.create_raw_dataset -i ~/path/to/common-voice --dataset-type common-voice \
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
        ArgOption(
            "--negative-pct", type=int, default=1, help="The percentage of the dataset to check for negative examples.",
        ),
        ArgOption(
            "--positive-pct",
            type=int,
            default=100,
            help="The percentage of the dataset to check for positive examples.",
        ),
        ArgOption("--input-audio-dataset-path", "-i", type=str, help="location of the input audio dataset",),
        ArgOption(
            "--dataset-type",
            type=str,
            default=AudioDatasetType.COMMON_VOICE.value,
            choices=[e.value for e in AudioDatasetType],
            help="type of dataset to use",
        ),
    )
    args = apb.parser.parse_args()

    dataset_type = AudioDatasetType(args.dataset_type)
    dataset_loader = get_dataset_loader(dataset_type, Path(args.input_audio_dataset_path))
    ds_kwargs = dict(sample_rate=SETTINGS.audio.sample_rate, mono=SETTINGS.audio.use_mono)
    train_ds, dev_ds, test_ds = dataset_loader.load_splits(**ds_kwargs)

    ctx = InferenceContext(vocab=SETTINGS.training.vocab, token_type=SETTINGS.training.token_type)
    if dataset_type == AudioDatasetType.COMMON_VOICE:
        train_ds = train_ds.filter(filter_fn)
        dev_ds = dev_ds.filter(filter_fn)
        test_ds = test_ds.filter(filter_fn)

    word_searcher = ctx.searcher if ctx.token_type == "word" else None
    for dataset in train_ds, dev_ds, test_ds:
        dataset.print_stats(word_searcher=word_searcher, compute_length=True)
        Logger.info(f"Generating {dataset.split.value} dataset")
        AudioDatasetWriter(dataset, AudioDatasetType.RAW).write(Path(SETTINGS.dataset.dataset_path))


if __name__ == "__main__":
    main()
