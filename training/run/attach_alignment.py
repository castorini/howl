from pathlib import Path

from howl.data.common.tokenizer import TokenType
from howl.dataset.aligned_audio_dataset_generator import AlignedAudioDatasetGenerator, AlignmentType
from howl.utils.args_utils import ArgOption, ArgumentParserBuilder


def main(
    input_raw_audio_dataset_path: Path, token_type: TokenType, alignment_type: AlignmentType, alignments_path: Path,
):
    """Attach alignments to the raw audio dataset and generate aligned-metadata

    Args:
        input_raw_audio_dataset_path: raw audio dataset of which alignments will be attached to
        alignment_type: type of alignment (MFA or STUB)
        token_type: type of token (WORD or PHONE)
        alignments_path (Optional): path of the dir which the MFA alignments are saved

    """
    aligned_dataset_generator = AlignedAudioDatasetGenerator(
        input_raw_audio_dataset_path, alignment_type, alignments_path, token_type=token_type
    )

    aligned_dataset_generator.generate_datasets()


def setup():
    """Parse the arguments"""

    apb = ArgumentParserBuilder()
    apb.add_options(
        ArgOption("--input-raw-audio-dataset-path", "-i", type=str, help="location of the input raw audio dataset",),
        ArgOption(
            "--token-type", type=str, choices=[e.value for e in TokenType], help="type of token (word or phone)",
        ),
        ArgOption(
            "--alignment-type",
            type=str,
            choices=[e.value for e in AlignmentType],
            help="type of alignment of the alignments",
        ),
        ArgOption("--alignments-path", type=str, help="location of the alignment files",),
    )
    raw_args = apb.parser.parse_args()

    if raw_args.alignment_type == AlignmentType.MFA and raw_args.alignments_path is None:
        raise ValueError("For MFA alignment type, alignments path must be provided")

    return raw_args


if __name__ == "__main__":
    args = setup()

    main(
        Path(args.input_raw_audio_dataset_path),
        TokenType(args.token_type),
        AlignmentType(args.alignment_type),
        Path(args.alignments_path) if args.alignments_path is not None else None,
    )
