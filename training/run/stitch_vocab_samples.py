import argparse
from pathlib import Path

from howl.data.common.labeler import WordFrameLabeler
from howl.data.common.searcher import WordTranscriptSearcher
from howl.data.common.vocab import Vocab
from howl.dataset.audio_dataset_constants import AudioDatasetType
from howl.dataset.stitched_audio_dataset_generator import StitchedAudioDatasetGenerator
from howl.dataset_loader.howl_audio_dataset_loader import HowlAudioDatasetLoader
from howl.settings import SETTINGS
from howl.utils.logger import Logger


def main():
    """Using aligned dataset, generate wakeword samples by stitching vocab samples

    VOCAB='["vocab1","vocab2","vocab3"]' INFERENCE_SEQUENCE=[1,2,3] \
        python -m training.run.stitch_vocab_samples \
        --dataset-path datasets/vocab1_vocab2_vocab3/positive \
        --stitched-dataset "stitched-dataset" \
        --max-num-training-samples 5000 \
        --max-num-dev-samples 1000 \
        --max-num-test-samples 1000

    The stitched metadata and audio files will be located under <dataset-path>
    """

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-path", type=str, help="input dataset for stitching; it must have frame labelled samples",
    )
    parser.add_argument(
        "--max-num-training-samples",
        type=int,
        default=5000,
        help="maximum number of training stitched samples to generate (default: 5000)",
    )
    parser.add_argument(
        "--max-num-dev-samples",
        type=int,
        default=1000,
        help="maximum number of dev stitched samples to generate (default: 1000)",
    )
    parser.add_argument(
        "--max-num-test-samples",
        type=int,
        default=1000,
        help="maximum number of test stitched samples to generate (default: 1000)",
    )
    parser.add_argument(
        "--disable-detect-keyword", action="store_false", help="disable keyword detection based verifcation",
    )

    args = parser.parse_args()

    vocab = Vocab(SETTINGS.training.vocab)
    if len(vocab) <= 1:
        Logger.warning(f"Word stitching require at least two words: {vocab}")
        return

    labeller = WordFrameLabeler(vocab)

    aligned_audio_dataset_path = Path(args.dataset_path)

    stitched_audio_dataset_generator = StitchedAudioDatasetGenerator(
        aligned_audio_dataset_path=aligned_audio_dataset_path,
        vocab=vocab,
        max_num_training_samples=args.max_num_training_samples,
        max_num_dev_samples=args.max_num_dev_samples,
        max_num_test_samples=args.max_num_test_samples,
        validate_stitched_sample=True,
        labeller=labeller,
    )

    stitched_audio_dataset_generator.generate_datasets()

    word_searcher = WordTranscriptSearcher(vocab)
    loader = HowlAudioDatasetLoader(AudioDatasetType.STITCHED, aligned_audio_dataset_path)
    for dataset in loader.load_splits():
        Logger.info(f"{dataset.set_type} stitched dataset: {dataset.compute_statistics(word_searcher=word_searcher)}")


if __name__ == "__main__":
    main()
