from tqdm import tqdm

from .args import ArgumentParserBuilder, opt
from ww4ff.data.dataset import MozillaKeywordLoader, AudioClipDatasetWriter, AudioClipDatasetMetadataWriter, \
    AlignedAudioClipMetadata
from ww4ff.align import LeftRightVadAligner
from ww4ff.settings import SETTINGS


def main():
    def filter_fn(x):
        return any(word in f' {x.transcription.lower()}' for word in args.target_words)

    apb = ArgumentParserBuilder()
    apb.add_options(opt('--target-words', type=str, nargs='+', default=[' hey', 'firefox']))
    args = apb.parser.parse_args()

    loader = MozillaKeywordLoader()
    ds_kwargs = dict(sr=SETTINGS.audio.sample_rate, mono=SETTINGS.audio.use_mono)
    cv_train_ds, cv_dev_ds, cv_test_ds = loader.load_splits(SETTINGS.raw_dataset.keyword_voice_dataset_path, **ds_kwargs)
    cv_train_ds.filter(filter_fn)
    cv_dev_ds.filter(filter_fn)
    cv_test_ds.filter(filter_fn)

    for ds in cv_dev_ds, cv_train_ds, cv_test_ds:
        AudioClipDatasetWriter(ds, 'a').write(SETTINGS.dataset.dataset_path)

        with AudioClipDatasetMetadataWriter(SETTINGS.dataset.dataset_path, ds.set_type, 'aligned-', mode='a') as writer:
            for ex in tqdm(ds, total=len(ds)):
                ex.metadata.transcription = ex.metadata.transcription.lower().replace('firefox', 'fire fox')  # TODO: remove quick fix
                aligned = LeftRightVadAligner().align(ex)
                writer.write(AlignedAudioClipMetadata(path=ex.metadata.path, transcription=aligned))


if __name__ == '__main__':
    main()
