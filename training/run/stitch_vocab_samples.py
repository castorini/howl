import argparse
from pathlib import Path

from howl.data.dataset import (
    AudioDatasetWriter,
    WakeWordDatasetLoader,
    WordFrameLabeler,
)
from howl.data.stitcher import WordStitcher
from howl.data.tokenize import Vocab
from howl.settings import SETTINGS

"""Using aligned dataset, generate wakeword samples by stitching vocab samples

VOCAB="[vocab1,vocab2,vocab3]" INFERENCE_SEQUENCE=[1,2,3] \
    python -m training.run.stitch_vocab_samples \
    --aligned-dataset "aligned-dataset" \
    --stitched-dataset "stitched-dataset" \
    --num-stitched-samples 500 \
    --stitched-dataset-pct [0.5, 0.25, 0.25]
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stitched-dataset",
        type=str,
        default="",
        help="if provided, stitched wakeword samples are saved to the given path. (default: dataset-path/stitched)",
    )
    parser.add_argument(
        "--aligned-dataset",
        type=str,
        help="dataset with frame labelled samples which stitched wakeword samples are generated from",
    )
    parser.add_argument(
        "--num-stitched-samples",
        type=int,
        default=10000,
        help="number of stitched wakeword samples to geneate (default: 10000)",
    )
    parser.add_argument(
        "--stitched-dataset-pct",
        type=int,
        nargs=3,
        default=[0.5, 0.25, 0.25],
        help="train/dev/test pct for stitched dataset (default: [0.5, 0.25, 0.25])",
    )
    parser.add_argument(
        "--disable-detect-keyword", action="store_false", help="disable keyword detection based verifcation"
    )

    args = parser.parse_args()
    aligned_ds_path = Path(args.aligned_dataset)
    stitched_ds_path = aligned_ds_path / "stitched" if args.stitched_dataset == "" else Path(args.stitched_dataset)
    stitched_ds_path.mkdir(exist_ok=True)

    vocab = Vocab(SETTINGS.training.vocab)
    labeler = WordFrameLabeler(vocab)
    ds_kwargs = dict(sr=SETTINGS.audio.sample_rate, mono=SETTINGS.audio.use_mono, frame_labeler=labeler)

    # load aligned datasets
    train_ds, dev_ds, test_ds = WakeWordDatasetLoader().load_splits(aligned_ds_path, **ds_kwargs)

    # stitch vocab samples
    stitcher = WordStitcher(vocab=vocab, detect_keyword=args.disable_detect_keyword)
    stitcher.stitch(args.num_stitched_samples, stitched_ds_path, train_ds, dev_ds, test_ds)

    # split the stitched samples
    stitched_train_ds, stitched_dev_ds, stitched_test_ds = stitcher.load_splits(*args.stitched_dataset_pct)

    # save metadata
    for ds in stitched_train_ds, stitched_dev_ds, stitched_test_ds:
        try:
            AudioDatasetWriter(ds, prefix="aligned-").write(stitched_ds_path)
        except KeyboardInterrupt:
            print("Skipping...")
            pass


if __name__ == "__main__":
    main()
