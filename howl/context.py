import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List

from howl.data.dataset import (
    PhonePhrase,
    PhoneticFrameLabeler,
    PronunciationDictionary,
    WakeWordDataset,
    WordFrameLabeler,
)
from howl.data.searcher import (
    LabelColoring,
    PhoneticTranscriptSearcher,
    WordTranscriptSearcher,
)
from howl.data.tokenize import Vocab
from howl.settings import SETTINGS


@dataclass
class WakewordDatasetContext:
    training_set: WakeWordDataset
    dev_positive_set: WakeWordDataset
    dev_negative_set: WakeWordDataset
    test_positive_set: WakeWordDataset
    test_negative_set: WakeWordDataset


class InferenceContext:
    def __init__(self, vocab: List[str], token_type: str = "phone", use_blank: bool = False):

        self.coloring = None
        self.adjusted_vocab = []
        self.num_labels = 0
        self.token_type = token_type
        self.pronounce_dict = None

        # break down each vocab into phonemes
        if token_type == "phone":
            self.pronounce_dict = PronunciationDictionary.from_file(Path(SETTINGS.training.phone_dictionary))

            self.coloring = LabelColoring()
            for word in vocab:
                # TODO:: we currently use single representation for simplicity
                phone_phrase = self.pronounce_dict.encode(word)[0]
                logging.info(f"Word {word: <10} has phonemes of {str(phone_phrase)}")
                self.add_vocab(list(str(phone) for phone in phone_phrase.phones))

        elif token_type == "word":
            self.add_vocab(vocab)

        # initialize vocab set for the system
        self.negative_label = len(self.adjusted_vocab)
        self.vocab = Vocab(
            {word: idx for idx, word in enumerate(self.adjusted_vocab)}, oov_token_id=self.negative_label
        )

        # initialize labeler; make sure this is located before adding other labels
        if token_type == "phone":
            phone_phrases = [PhonePhrase.from_string(x) for x in self.adjusted_vocab]
            self.labeler = PhoneticFrameLabeler(self.pronounce_dict, phone_phrases)
        elif token_type == "word":
            self.labeler = WordFrameLabeler(self.vocab)

        # add negative label
        self.add_vocab(["[OOV]"])

        # initialize TranscriptSearcher with the processed targets
        if token_type == "phone":
            self.searcher = PhoneticTranscriptSearcher(phone_phrases, self.coloring)
        elif token_type == "word":
            self.searcher = WordTranscriptSearcher(self.vocab)

        # add extra label for blank if necessary
        self.blank_label = -1
        if use_blank:
            self.blank_label = len(self.adjusted_vocab)
            self.add_vocab(["[BLANK]"])

        for idx, word in enumerate(self.adjusted_vocab):
            logging.info(f"target {word:10} is assigned to label {idx}")

    def add_vocab(self, vocabs: List[str]):
        for vocab in vocabs:
            self.adjusted_vocab.append(vocab)
        if self.coloring:
            self.coloring.extend_sequence(len(vocabs))
        self.num_labels += len(vocabs)
