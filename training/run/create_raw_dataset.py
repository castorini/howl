import logging
from pathlib import Path

from howl.context import InferenceContext
from howl.data.dataset import (AudioDataset, AudioDatasetWriter,
                               RegisteredPathDatasetLoader)
from howl.settings import SETTINGS
from howl.utils.hash import sha256_int

from .args import ArgumentParserBuilder, opt

"""
This scripts processes given audio dataset and creates datasets that howl can take in
distributions can be customized using negative-pct and positive-pct arguments

sample command:
VOCAB='["fire"]' INFERENCE_SEQUENCE=[0] DATASET_PATH=data/fire-positive \
python -m training.run.create_raw_dataset -i ~/path/to/common-voice --positive-pct 100 --negative-pct 0
"""


def print_stats(header: str, context: InferenceContext, * datasets: AudioDataset, compute_length=True):
    word_searcher = context.searcher if context.token_type == 'word' else None
    for ds in datasets:
        logging.info(
            f'{header} ({ds.set_type}) statistics: {ds.compute_statistics(word_searcher=word_searcher, compute_length=compute_length)}')


def main():
    def filter_fn(x):
        bucket = sha256_int(x.path.stem) % 100
        if bucket < args.negative_pct:
            # drop the samples whose transcript is wakeword
            return not ctx.searcher.search(x.transcription.lower())
        if bucket < args.positive_pct:
            # select all sample whose transcript contains at least one of the vocabs
            return ctx.searcher.contains_any(x.transcription.lower())
        return False

    apb = ArgumentParserBuilder()
    apb.add_options(opt('--negative-pct',
                        type=int,
                        default=1,
                        help='The percentage of the dataset to check for negative examples.'),
                    opt('--positive-pct', type=int, default=100,
                        help='The percentage of the dataset to check for positive examples.'),
                    opt('--input-path', '-i', type=str),
                    opt('--dataset-type',
                        type=str,
                        default='mozilla-cv',
                        choices=RegisteredPathDatasetLoader.registered_names()))
    args = apb.parser.parse_args()
    if args.input_path is None:
        args.input_path = SETTINGS.raw_dataset.common_voice_dataset_path

    ctx = InferenceContext(SETTINGS.training.vocab,
                           token_type=SETTINGS.training.token_type)
    loader = RegisteredPathDatasetLoader.find_registered_class(
        args.dataset_type)()
    ds_kwargs = dict(sr=SETTINGS.audio.sample_rate,
                     mono=SETTINGS.audio.use_mono)
    train_ds, dev_ds, test_ds = loader.load_splits(
        Path(args.input_path), **ds_kwargs)

    if args.dataset_type == 'mozilla-cv':
        train_ds = train_ds.filter(filter_fn)
        dev_ds = dev_ds.filter(filter_fn)
        test_ds = test_ds.filter(filter_fn)

    print_stats('Dataset', ctx, train_ds, dev_ds, test_ds, compute_length=True)

    for ds in train_ds, dev_ds, test_ds:
        try:
            AudioDatasetWriter(ds).write(Path(SETTINGS.dataset.dataset_path))
        except KeyboardInterrupt:
            logging.info('Skipping...')
            pass


if __name__ == '__main__':
    main()
