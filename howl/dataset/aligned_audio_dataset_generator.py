import functools
import logging
import multiprocessing
from enum import Enum, unique
from pathlib import Path
from typing import Dict, Tuple

from textgrids import TextGrid
from tqdm import tqdm

from howl.data.common.metadata import AudioClipMetadata
from howl.data.common.tokenizer import TokenType
from howl.data.dataset.dataset_writer import AudioDatasetMetadataWriter
from howl.dataset.raw_audio_dataset import RawAudioDataset
from howl.dataset_loader.raw_audio_dataset_loader import RawAudioDatasetLoader
from howl.settings import SETTINGS
from howl.utils import logging_utils
from training.align import MfaTextGridConverter, StubAligner
from training.align.base import AlignedTranscription


@unique
class AlignmentType(str, Enum):
    """String based Enum for alignment type"""

    MFA = "mfa"
    STUB = "stub"


class AlignedAudioDatasetGenerator:
    """Using raw audio dataset and the alignment generated with MFA, generate new set of json metadata json files"""

    ALIGNED_METADATA_PREFIX = "aligned-"

    def __init__(
        self,
        raw_audio_dataset_path: Path,
        alignment_type: AlignmentType,
        alignments_path: Path = None,
        sample_rate: int = SETTINGS.audio.sample_rate,
        mono: bool = SETTINGS.audio.use_mono,
        token_type: TokenType = TokenType(SETTINGS.training.token_type),
        logger: logging.Logger = None,
    ):
        """initialize AlignedAudioDatasetGenerator by loading dataset from the given path

        Args:
            raw_audio_dataset_path: location of the dataset
            alignment_type: type of the alignment
            alignments_path: location of the alignments
            mono: if True, the audio file will be loaded assuming the data is mono channel
            sample_rate: sample rate of the audio file
            logger: logger
        """

        self.logger = logger
        if self.logger is None:
            self.logger = logging_utils.setup_logger(self.__class__.__name__)

        self.raw_audio_dataset_path = raw_audio_dataset_path
        if not self.raw_audio_dataset_path.exists():
            raise FileNotFoundError(f"Dataset path is invalid: {self.raw_audio_dataset_path}")

        raw_audio_dataset = RawAudioDatasetLoader(self.raw_audio_dataset_path, self.logger)
        ds_kwargs = dict(sample_rate=sample_rate, mono=mono)
        self.train_ds, self.dev_ds, self.test_ds = raw_audio_dataset.load_splits(**ds_kwargs)

        if alignment_type == AlignmentType.MFA:
            self.alignments = self._load_mfa_alignments(alignments_path, token_type)
        elif alignment_type == AlignmentType.STUB:
            self.alignments = {}
            self.alignments.update(self._load_stub_alignments(self.train_ds))
            self.alignments.update(self._load_stub_alignments(self.dev_ds))
            self.alignments.update(self._load_stub_alignments(self.test_ds))
        else:
            raise ValueError(f"Alignment type is invalid: {alignment_type}")

    @staticmethod
    def _load_mfa_alignment(alignment_file_path: Path, use_phone: bool) -> Tuple[str, AlignedTranscription]:
        """Helper function which loads single file of MFA alignment

        Args:
            alignment_file_path: MFA alignment file (TextGrid)
            use_phone: True to load phone-based alignment

        Returns:
            audio_id and AlignedTranscription instance
        """
        converter = MfaTextGridConverter(use_phones=use_phone)
        text_grid = TextGrid(str(alignment_file_path.absolute()))
        audio_id = alignment_file_path.stem
        alignment = converter.convert(text_grid)
        return audio_id, alignment

    def _load_mfa_alignments(self, alignments_path: Path, token_type: TokenType) -> Dict[str, AlignedTranscription]:
        """Loads all the MFA alignments in memory

        Args:
            alignments_path: directory that contains all the alignment file (TextGrid)
            token_type: type of alignment to load

        Returns:
            Mapping from audio id to alignments
        """
        num_processes = max(multiprocessing.cpu_count() // 2, 4)
        pool = multiprocessing.Pool(processes=num_processes)

        alignment_file_paths = list(alignments_path.glob("**/*.TextGrid"))
        alignment_pair_list = tqdm(
            pool.imap(
                functools.partial(
                    AlignedAudioDatasetGenerator._load_mfa_alignment, use_phone=(token_type == TokenType.PHONE.value)
                ),
                alignment_file_paths,
            ),
            desc=f"loading alignments from {alignments_path}",
            total=(len(alignment_file_paths)),
        )

        alignments: Dict[str, AlignedTranscription] = {}
        for audio_id, alignment in alignment_pair_list:
            alignments[audio_id] = alignment

        return alignments

    @staticmethod
    def _load_stub_alignment(metadata: AudioClipMetadata, sample_rate: int, mono: bool):
        """Helper function which Generate STUB alignments for each sample

        Args:
            metadata: Metadata of the audio sample which STUB alignment will be generated for
            sample_rate: Sample rate of the audio data
            mono: True to load only mono-channel of audio data

        Returns:
            audio_id and AlignedTranscription instance
        """
        sample = RawAudioDataset.load_sample(metadata, sample_rate, mono)
        return sample.metadata.audio_id, StubAligner().align(sample)

    def _load_stub_alignments(self, raw_audio_dataset: RawAudioDataset):
        """Loads STUB alignments for the given raw audio dataset

        Args:
            raw_audio_dataset: Raw audio dataset which the STUB alignments will be generated for

        Returns:
            Mapping from audio id to alignments
        """
        num_processes = max(multiprocessing.cpu_count() // 2, 4)
        pool = multiprocessing.Pool(processes=num_processes)

        alignment_pair_list = tqdm(
            pool.imap(
                functools.partial(
                    AlignedAudioDatasetGenerator._load_stub_alignment,
                    sample_rate=raw_audio_dataset.sample_rate,
                    mono=raw_audio_dataset.mono,
                ),
                raw_audio_dataset.metadata_list,
            ),
            desc=f"loading alignments for {raw_audio_dataset}",
            total=(len(raw_audio_dataset)),
        )

        alignments = {}
        for audio_id, alignment in alignment_pair_list:
            alignments[audio_id] = alignment

        return alignments

    @staticmethod
    def _generate_metadata_with_alignment(
        metadata: AudioClipMetadata, alignments: Dict[str, AlignedTranscription], logger: logging.Logger
    ):
        """Helper function which attaches alignment to the given sample

        Args:
            metadata: Metadata of the audio sample which alignment will be attached to
            alignments: Map of alignments
            logger: logger

        Returns:
            AudioClipMetadata with alignment
        """
        if metadata.audio_id not in alignments:
            logger.warning(f"Alignments for audio file {metadata.audio_id} does not exist")
            return None
        aligned_transcription = alignments[metadata.audio_id]
        return AudioClipMetadata(
            path=metadata.path,
            transcription=aligned_transcription.transcription,
            end_timestamps=aligned_transcription.end_timestamps,
        )

    def _generate_dataset(self, raw_audio_dataset: RawAudioDataset):
        """Transform given raw audio dataset into aligned audio dataset by generating aligned metadata file

        Args:
            raw_audio_dataset: raw audio dataset to transform
        """
        num_processes = max(multiprocessing.cpu_count() // 2, 4)
        pool = multiprocessing.Pool(processes=num_processes)

        metadata_list = tqdm(
            pool.imap(
                functools.partial(
                    AlignedAudioDatasetGenerator._generate_metadata_with_alignment,
                    alignments=self.alignments,
                    logger=self.logger,
                ),
                raw_audio_dataset.metadata_list,
            ),
            desc=f"generating aligned metadata for {raw_audio_dataset}",
            total=(len(raw_audio_dataset)),
        )

        metadata_list = list(filter(None, metadata_list))  # remove None entries

        with AudioDatasetMetadataWriter(
            self.raw_audio_dataset_path, raw_audio_dataset.dataset_split, prefix=self.ALIGNED_METADATA_PREFIX
        ) as metadata_writer:
            for metadata in metadata_list:
                metadata_writer.write(metadata)

    def generate_datasets(self):
        """Transform raw audio datasets into aligned audio datasets by generating aligned metadata files"""
        self._generate_dataset(self.train_ds)
        self._generate_dataset(self.dev_ds)
        self._generate_dataset(self.test_ds)
