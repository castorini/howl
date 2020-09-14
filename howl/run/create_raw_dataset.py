from pathlib import Path
import logging

from .args import ArgumentParserBuilder, opt
from howl.context import InferenceContext
from howl.data.dataset import AudioDatasetWriter, AudioDataset, RegisteredPathDatasetLoader
from howl.settings import SETTINGS
from howl.utils.hash import sha256_int


def print_stats(header: str, *datasets: AudioDataset, skip_length=False):
    for ds in datasets:
        logging.info(f'{header} ({ds.set_type}) statistics: {ds.compute_statistics(skip_length=skip_length)}')


def main():
    def filter_fn(x):
        bucket = sha256_int(x.path.stem) % 100
        if bucket < args.negative_pct:
            return not ctx.searcher.search(x.transcription.lower())
        if bucket < args.positive_pct:
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


    print(args.dataset_type)

    ctx = InferenceContext(SETTINGS.training.vocab, token_type=SETTINGS.training.token_type)
    loader = RegisteredPathDatasetLoader.find_registered_class(args.dataset_type)()
    ds_kwargs = dict(sr=SETTINGS.audio.sample_rate, mono=SETTINGS.audio.use_mono)
    # cv_train_ds, cv_dev_ds, cv_test_ds = loader.load_splits(Path(args.input_path), **ds_kwargs)

    # if args.dataset_type == 'mozilla-cv':
    #     cv_train_ds = cv_train_ds.filter(filter_fn)
    #     cv_dev_ds = cv_dev_ds.filter(filter_fn)
    #     cv_test_ds = cv_test_ds.filter(filter_fn)
        
    # print_stats('Dataset', cv_train_ds, cv_dev_ds, cv_test_ds, skip_length=True)

    # for ds in cv_train_ds, cv_dev_ds, cv_test_ds:
    #     try:
    #         AudioDatasetWriter(ds).write(Path(SETTINGS.dataset.dataset_path))
    #     except KeyboardInterrupt:
    #         logging.info('Skipping...')
    #         pass

    def train_set():
        cv_train_ds = loader.load_train_set(Path(args.input_path), **ds_kwargs)
        AudioDatasetWriter(cv_train_ds).write(Path(SETTINGS.dataset.dataset_path))

    def dev_set():
        cv_dev_ds = loader.load_dev_set(Path(args.input_path), **ds_kwargs)
        AudioDatasetWriter(cv_dev_ds).write(Path(SETTINGS.dataset.dataset_path))

    def test_set():
        cv_test_ds = loader.load_test_set(Path(args.input_path), **ds_kwargs)
        AudioDatasetWriter(cv_test_ds).write(Path(SETTINGS.dataset.dataset_path))

    train_set()
    dev_set()
    test_set()


if __name__ == '__main__':
    main()
