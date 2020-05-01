import logging

from .args import ArgumentParserBuilder, opt
from ww4ff.data.dataset import MozillaWakeWordLoader, MozillaCommonVoiceLoader, FlatWavDatasetWriter, AudioClipDataset
from ww4ff.settings import SETTINGS
from ww4ff.utils.hash import sha256_int


def print_stats(header: str, *datasets: AudioClipDataset, skip_length=False):
    for ds in datasets:
        logging.info(f'{header} ({ds.set_type}) statistics: {ds.compute_statistics(skip_length=skip_length)}')


def main():
    def filter_fn(x):
        return sha256_int(x.path.stem) % 100 < args.filter_pct

    apb = ArgumentParserBuilder()
    apb.add_options(opt('--filter-pct', type=int, default=1, help='The percentage of the Common Voice dataset to use.'),
                    opt('--split-type',
                        type=str,
                        default='sound',
                        choices=('sound', 'speaker'),
                        help='Split by sound ID or speaker ID.'))
    args = apb.parser.parse_args()

    loader = MozillaCommonVoiceLoader()
    ds_kwargs = dict(sr=SETTINGS.audio.sample_rate, mono=SETTINGS.audio.use_mono)
    cv_train_ds, cv_dev_ds, cv_test_ds = loader.load_splits(SETTINGS.raw_dataset.common_voice_dataset_path, **ds_kwargs)
    cv_train_ds = cv_train_ds.filter(filter_fn)
    cv_dev_ds = cv_dev_ds.filter(filter_fn)
    cv_test_ds = cv_test_ds.filter(filter_fn)
    print_stats('Filtered Common Voice dataset', cv_train_ds, cv_dev_ds, cv_test_ds, skip_length=True)

    loader = MozillaWakeWordLoader(split_by_speaker=args.split_type == 'speaker')
    ww_train_ds, ww_dev_ds, ww_test_ds = loader.load_splits(SETTINGS.raw_dataset.wake_word_dataset_path, **ds_kwargs)
    print_stats('Wake word dataset', ww_train_ds, ww_dev_ds, ww_test_ds, skip_length=True)

    ww_train_ds.extend(cv_train_ds)
    ww_dev_ds.extend(cv_dev_ds)
    ww_test_ds.extend(cv_test_ds)

    loader = MozillaWakeWordLoader(split_by_speaker=args.split_type == 'speaker', split='rejected')
    rej_train_ds, rej_dev_ds, rej_test_ds = loader.load_splits(SETTINGS.raw_dataset.wake_word_dataset_path, **ds_kwargs)
    print_stats('Rejected dataset', rej_train_ds, rej_dev_ds, rej_test_ds, skip_length=True)

    ww_train_ds.extend(rej_train_ds)
    ww_dev_ds.extend(rej_dev_ds)
    ww_test_ds.extend(rej_test_ds)
    print_stats('Combined dataset', ww_train_ds, ww_dev_ds, ww_test_ds, skip_length=True)

    for ds in ww_train_ds, ww_dev_ds, ww_test_ds:
        FlatWavDatasetWriter(ds).write(SETTINGS.dataset.dataset_path)


if __name__ == '__main__':
    main()
