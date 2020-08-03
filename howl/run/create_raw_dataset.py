from pathlib import Path
import logging

from .args import ArgumentParserBuilder, opt
from howl.data.dataset import AudioDatasetWriter, AudioDataset, RegisteredPathDatasetLoader
from howl.model.inference import InferenceEngineSettings
from howl.settings import SETTINGS
from howl.utils.hash import sha256_int


def print_stats(header: str, *datasets: AudioDataset, skip_length=False):
    for ds in datasets:
        logging.info(f'{header} ({ds.set_type}) statistics: {ds.compute_statistics(skip_length=skip_length)}')


def main():
    def filter_fn(x):
        bucket = sha256_int(x.path.stem) % 100
        if bucket < args.negative_pct:
            return wake_word not in f' {x.transcription.lower()} '
        if bucket < args.positive_pct:
            return any(word in f' {x.transcription.lower()} ' for word in args.vocab)
        return False

    apb = ArgumentParserBuilder()
    apb.add_options(opt('--negative-pct',
                        type=int,
                        default=1,
                        help='The percentage of the dataset to check for negative examples.'),
                    opt('--vocab', type=str, nargs='+', default=[' hey', 'fire', 'fox']),
                    opt('--positive-pct', type=int, default=100,
                        help='The percentage of the dataset to check for positive examples.'),
                    opt('--input-path', '-i', type=Path),
                    opt('--dataset-type',
                        type=str,
                        default='mozilla-cv',
                        choices=RegisteredPathDatasetLoader.registered_names()))
    args = apb.parser.parse_args()
    if args.input_path is None:
        args.input_path = SETTINGS.raw_dataset.common_voice_dataset_path

    wake_word = InferenceEngineSettings().make_wakeword(args.vocab)
    loader = RegisteredPathDatasetLoader.find_registered_class(args.dataset_type)()
    ds_kwargs = dict(sr=SETTINGS.audio.sample_rate, mono=SETTINGS.audio.use_mono)
    cv_train_ds, cv_dev_ds, cv_test_ds = loader.load_splits(args.input_path, **ds_kwargs)
    cv_train_ds = cv_train_ds.filter(filter_fn)
    cv_dev_ds = cv_dev_ds.filter(filter_fn)
    cv_test_ds = cv_test_ds.filter(filter_fn)
    print_stats('Dataset', cv_train_ds, cv_dev_ds, cv_test_ds, skip_length=True)

    for ds in cv_train_ds, cv_dev_ds, cv_test_ds:
        try:
            AudioDatasetWriter(ds).write(SETTINGS.dataset.dataset_path)
        except KeyboardInterrupt:
            logging.info('Skipping...')
            pass


if __name__ == '__main__':
    main()
