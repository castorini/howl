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
        self.labeler = WordFrameLabeler(vocab)
        self.coloring = None
        use_phone = token_type == 'phone'
        if pronounce_dict is None and use_phone:
            pronounce_dict = PronunciationDictionary.from_file(SETTINGS.training.phone_dictionary)
        if use_phone:
            self.coloring = LabelColoring()
            new_vocab = []
            if use_blank:
                new_vocab.append('[BLANK]')
                self.coloring.label_counter = 1
            for word in vocab:
                phone_phrases = pronounce_dict.encode(word)
                logging.info(f'Using phonemes {str(phone_phrases)} for word {word}.')
                new_vocab.extend(x.text for x in phone_phrases)
                self.coloring.extend_sequence(len(phone_phrases))
            self.coloring.extend_sequence(1)  # negative label
            if use_blank:
                self.coloring.append_label(0)  # blank
            vocab = new_vocab
            phone_phrases = [PhonePhrase.from_string(x) for x in new_vocab]
            self.labeler = PhoneticFrameLabeler(phone_phrases)
        self.num_labels = len(vocab) + 1 + use_blank
        self.vocab = Vocab({word: idx for idx, word in enumerate(vocab)})
        self.negative_label = len(vocab)
        self.searcher = PhoneticTranscriptSearcher(phone_phrases, self.coloring) if use_phone else WordTranscriptSearcher(vocab)
