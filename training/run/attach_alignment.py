from itertools import chain
from pathlib import Path
import argparse

from textgrids import TextGrid
from tqdm import tqdm

from training.align import MfaTextGridConverter, StubAligner
from howl.data.dataset import AudioClipDatasetLoader, AudioDatasetMetadataWriter
from howl.data.dataset.base import AudioClipMetadata as AlignedAudioClipMetadata
from howl.settings import SETTINGS


def main():
    def load_mfa_align():
        converter = MfaTextGridConverter(use_phones=SETTINGS.training.token_type == 'phone')
        id_align_map = {}

        for tg_path in args.align_folder.glob('**/*.TextGrid'):
            tg = TextGrid(str(tg_path.absolute()))
            audio_id = tg_path.name.split('.', 1)[0]
            id_align_map[audio_id] = converter.convert(tg)
        return id_align_map

    def load_stub_align():
        id_align_map = {}
        for ex in tqdm(chain(train_ds, dev_ds, test_ds), total=sum(map(len, (train_ds, dev_ds, test_ds)))):
            id_align_map[ex.metadata.audio_id] = StubAligner().align(ex)
        return id_align_map

    parser = argparse.ArgumentParser()
    parser.add_argument('--mfa-folder', '-i', dest='align_folder', type=Path)
    parser.add_argument('--align-type', type=str, default='mfa', choices=('mfa', 'stub'))
    args = parser.parse_args()

    ds_kwargs = dict(sr=SETTINGS.audio.sample_rate, mono=SETTINGS.audio.use_mono)
    ds_path = Path(SETTINGS.dataset.dataset_path)
    train_ds, dev_ds, test_ds = AudioClipDatasetLoader().load_splits(ds_path, **ds_kwargs)

    if args.align_type == 'mfa':
        id_align_map = load_mfa_align()
    elif args.align_type == 'stub':
        id_align_map = load_stub_align()
    else:
        raise ValueError

    for ds in (train_ds, dev_ds, test_ds):
        with AudioDatasetMetadataWriter(ds_path, ds.set_type, 'aligned-', mode='w') as writer:
            for ex in tqdm(ds, total=len(ds)):
                try:
                    transcription = id_align_map[ex.metadata.audio_id]
                    writer.write(AlignedAudioClipMetadata(path=ex.metadata.path,
                                                          transcription=transcription.transcription,
                                                          end_timestamps=transcription.end_timestamps))
                except KeyError:
                    pass


if __name__ == '__main__':
    main()
