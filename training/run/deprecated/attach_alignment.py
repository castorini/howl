import argparse
from itertools import chain
from pathlib import Path

from textgrids import TextGrid
from tqdm import tqdm

from howl.data.common.metadata import AudioClipMetadata
from howl.data.dataset.dataset_loader import AudioClipDatasetLoader
from howl.data.dataset.dataset_writer import AudioDatasetMetadataWriter
from howl.dataset.audio_dataset_constants import AudioDatasetType
from howl.settings import SETTINGS
from training.align import MfaTextGridConverter, StubAligner


def main():
    """Attach alignment to the raw audio dataset"""

    def load_mfa_align():
        """Load alignment using montreal forced aligner"""
        converter = MfaTextGridConverter(use_phones=SETTINGS.training.token_type == "phone")
        id_align_map = {}

        for tg_path in args.align_folder.glob("**/*.TextGrid"):
            text_grid = TextGrid(str(tg_path.absolute()))
            audio_id = tg_path.name.split(".", 1)[0]
            id_align_map[audio_id] = converter.convert(text_grid)
        return id_align_map

    def load_stub_align():
        """Load alignment for stub"""
        id_align_map = {}
        for ex in tqdm(chain(train_ds, dev_ds, test_ds), total=sum(map(len, (train_ds, dev_ds, test_ds))),):
            id_align_map[ex.metadata.audio_id] = StubAligner().align(ex)
        return id_align_map

    parser = argparse.ArgumentParser()
    parser.add_argument("--mfa-folder", "-i", dest="align_folder", type=Path)
    parser.add_argument("--align-type", type=str, default="mfa", choices=("mfa", "stub"))
    args = parser.parse_args()

    ds_kwargs = dict(sample_rate=SETTINGS.audio.sample_rate, mono=SETTINGS.audio.use_mono)
    ds_path = Path(SETTINGS.dataset.dataset_path)
    train_ds, dev_ds, test_ds = AudioClipDatasetLoader().load_splits(ds_path, **ds_kwargs)

    if args.align_type == "mfa":
        id_align_map = load_mfa_align()
    elif args.align_type == "stub":
        id_align_map = load_stub_align()
    else:
        raise ValueError

    for dataset in (train_ds, dev_ds, test_ds):
        with AudioDatasetMetadataWriter(ds_path, AudioDatasetType.ALIGNED, dataset.set_type) as writer:
            for ex in tqdm(dataset, total=len(dataset)):
                try:
                    transcription = id_align_map[ex.metadata.audio_id]
                    writer.write(
                        AudioClipMetadata(
                            path=ex.metadata.path,
                            transcription=transcription.transcription,
                            end_timestamps=transcription.end_timestamps,
                        )
                    )
                except KeyError:
                    pass


if __name__ == "__main__":
    main()
