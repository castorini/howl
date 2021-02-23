from dataclasses import dataclass
from typing import List
import logging

from howl.data.dataset import WordFrameLabeler, PhoneticFrameLabeler, PronunciationDictionary, PhonePhrase
from howl.model.inference import LabelColoring, PhoneticTranscriptSearcher, WordTranscriptSearcher
from howl.settings import SETTINGS
from howl.data.dataset import WakeWordDataset
from howl.data.tokenize import Vocab


@dataclass
class WakewordDatasetContext:
    training_set: WakeWordDataset
    dev_positive_set: WakeWordDataset
    dev_negative_set: WakeWordDataset
    test_positive_set: WakeWordDataset
    test_negative_set: WakeWordDataset


class InferenceContext:
    def __init__(self,
                 vocab: List[str],
                 token_type: str = 'phone',
                 pronounce_dict: PronunciationDictionary = None,
                 use_blank: bool = False):

        self.coloring = None
        self.adjusted_vocab = []
        self.num_labels = 0

        # break down each vocab into phonemes
        if token_type == 'phone':
            if pronounce_dict is None:
                pronounce_dict = PronunciationDictionary.from_file(
                    SETTINGS.training.phone_dictionary)

            self.coloring = LabelColoring()
            for word in vocab:
                phone_phrases = pronounce_dict.encode(word)
                logging.info(
                    f'Using phonemes {str(phone_phrases)} for word {word}')
                self.add_vocab(x.text for x in phone_phrases)

        elif token_type == 'word':
            self.add_vocab(vocab)

        # initialize labeler; make sure this is located before adding other labels
        if token_type == 'phone':
            phone_phrases = [PhonePhrase.from_string(
                x) for x in self.adjusted_vocab]
            self.labeler = PhoneticFrameLabeler(phone_phrases)
        elif token_type == 'word':
            print('labeler vocab: ', self.adjusted_vocab)
            self.labeler = WordFrameLabeler(self.adjusted_vocab)

        # initialize vocab set for the system and add negative label
        self.negative_label = len(self.adjusted_vocab)
        self.vocab = Vocab({word: idx for idx, word in enumerate(
            self.adjusted_vocab)}, oov_token_id=self.negative_label)
        self.add_vocab(['[OOV]'])

        # initialize TranscriptSearcher with the processed targets
        if token_type == 'phone':
            self.searcher = PhoneticTranscriptSearcher(
                phone_phrases, self.coloring)
        elif token_type == 'word':
            self.searcher = WordTranscriptSearcher(self.adjusted_vocab)

        # add extra label for blank if necessary
        self.blank_label = -1
        if use_blank:
            self.blank_label = len(self.adjusted_vocab)
            self.add_vocab(['[BLANK]'])

        for idx, word in enumerate(self.adjusted_vocab):
            logging.info(f'target {word:10} is assigned to label {idx}')

    def add_vocab(vocabs: List[str]):
        for vocab in vocabs:
            self.adjusted_vocab.append(vocab)
        if self.coloring:
            self.coloring.extend_sequence(len(vocabs))
        self.num_labels += len(vocabs)
