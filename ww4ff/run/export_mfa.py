from pathlib import Path
import argparse

from tqdm import tqdm

from ww4ff.data.dataset import AudioClipDatasetLoader
from ww4ff.settings import SETTINGS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-folder', '-o', type=Path, required=True)
    args = parser.parse_args()

    args.output_folder.mkdir(exist_ok=True, parents=True)
    ds_kwargs = dict(sr=SETTINGS.audio.sample_rate, mono=SETTINGS.audio.use_mono)
    train_ds, dev_ds, test_ds = AudioClipDatasetLoader().load_splits(SETTINGS.dataset.dataset_path, **ds_kwargs)
    for ds in (train_ds, dev_ds, test_ds):
        for metadata in tqdm(ds.metadata_list):
            lab_name = metadata.path.with_suffix('.lab').name
            with (args.output_folder / lab_name).open('w') as f:
                f.write(f'{metadata.transcription}\n')


if __name__ == '__main__':
    main()
