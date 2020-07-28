import logging

from tqdm import tqdm

from .args import ArgumentParserBuilder, opt
from ww4ff.data.dataset import MozillaWakeWordLoader, MozillaCommonVoiceLoader, AudioDatasetWriter, AudioClipDataset,\
    AlignedAudioClipMetadata, AudioDatasetMetadataWriter, AudioClipDatasetLoader
from ww4ff.align import StubAligner
from ww4ff.settings import SETTINGS
from ww4ff.utils.hash import sha256_int


def print_stats(header: str, *datasets: AudioClipDataset, skip_length=False):
    for ds in datasets:
        logging.info(f'{header} ({ds.set_type}) statistics: {ds.compute_statistics(skip_length=skip_length)}')


def main():
    def filter_fn(x):
        bucket = sha256_int(x.path.stem) % 100
        return bucket < args.filter_pct

    apb = ArgumentParserBuilder()
    apb.add_options(opt('--filter-pct', type=int, default=1, help='The percentage of the Common Voice dataset to use.'),
                    opt('--target-words', type=str, nargs='+', default=[' hey', 'fire', 'fox']),
                    opt('--split-type',
                        type=str,
                        default='speaker',
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

    for ds in cv_train_ds, cv_dev_ds, cv_test_ds:
        AudioDatasetWriter(ds, mode='a').write(SETTINGS.dataset.dataset_path)

    ds_path = SETTINGS.dataset.dataset_path
    aligner = StubAligner()
    train_ds, dev_ds, test_ds = AudioClipDatasetLoader().load_splits(ds_path, **ds_kwargs)
    for ds in (train_ds, dev_ds, test_ds):
        with AudioDatasetMetadataWriter(ds_path, ds.set_type, 'aligned-', mode='w') as writer:
            for ex in tqdm(ds, total=len(ds)):
                try:
                    transcription = aligner.align(ex)
                    writer.write(AlignedAudioClipMetadata(path=ex.metadata.path, transcription=transcription))
                except KeyError:
                    pass


if __name__ == '__main__':
    main()
