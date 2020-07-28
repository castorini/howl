from pathlib import Path
import argparse

from textgrids import TextGrid
from tqdm import tqdm

from ww4ff.align import MfaTextGridConverter
from ww4ff.data.dataset import AudioClipDatasetLoader, AudioDatasetMetadataWriter, WakeWordDatasetLoader, \
    AlignedAudioClipMetadata
from ww4ff.settings import SETTINGS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-folder', '-i', dest='align_folder', type=Path, required=True)
    args = parser.parse_args()

    converter = MfaTextGridConverter()
    ds_kwargs = dict(sr=SETTINGS.audio.sample_rate, mono=SETTINGS.audio.use_mono, words=[])
    ds_path = SETTINGS.dataset.dataset_path
    train_ds, dev_ds, test_ds = AudioClipDatasetLoader().load_splits(ds_path, **ds_kwargs)
    id_align_map = {}

    for tg_path in args.align_folder.glob('*.TextGrid'):
        tg = TextGrid(str(tg_path.absolute()))
        audio_id = tg_path.name.split('.', 1)[0]
        id_align_map[audio_id] = converter.convert(tg)

    for ds in (train_ds, dev_ds, test_ds):
        with AudioDatasetMetadataWriter(ds_path, ds.set_type, 'aligned-', mode='w') as writer:
            for ex in tqdm(ds, total=len(ds)):
                try:
                    transcription = id_align_map[ex.metadata.path.name.split('.', 1)[0]]
                    writer.write(AlignedAudioClipMetadata(path=ex.metadata.path, transcription=transcription))
                except KeyError:
                    pass


if __name__ == '__main__':
    main()
