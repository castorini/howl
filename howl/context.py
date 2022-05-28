import logging
from pathlib import Path
from typing import List

from howl.config import ContextConfig
from howl.data.common.labeler import PhoneticFrameLabeler, WordFrameLabeler
from howl.data.common.phone import PhonePhrase, PronunciationDictionary
from howl.data.common.searcher import LabelColoring, PhoneticTranscriptSearcher, WordTranscriptSearcher
from howl.data.common.tokenizer import TokenType
from howl.data.common.vocab import Vocab
from howl.settings import SETTINGS


class InferenceContext:
    """Basic configuration for the whole system"""

    def __init__(
        self,
        vocab: List[str],
        sequence: List[int] = None,
        token_type: str = TokenType.PHONE,
        phone_dictionary_path: str = None,
        seed: int = 0,
        use_blank: bool = False,
    ):
        """Initialize context by creating vocabs, label mappings, and searcher

        Args:
            vocab (List[str]): list of words / phonemes to detect
            sequence (List[int]): list of vocab index that makes up the wake word
                if None, the list of vocab would be the target wake word
            token_type (str): type of the token that the system will be based on
                word: training and inference will be achieved at word level
                phone: training and inference will be achieved at phoneme level
            phone_dictionary_path (str): phone dictionary file path
            seed (int): random seed
            use_blank (bool): if True, [BLANK] token will be added to vocab
                TODO: use_blank seems to be only necessary for the model with CTC loss
                      clear docstring and variable name would be recommended
        """

        self.seed = seed
        self.sequence = sequence
        if self.sequence is None:
            self.sequence = range(len(vocab))
        self.phone_dictionary_path = phone_dictionary_path

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
            self.labeler = PhoneticFrameLabeler(phone_phrases, self.pronounce_dict)
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
        """Add a list of vocab to vocab instance and coloring instance

        Args:
            vocabs (List[str]): list of vocab to append
        """
        for vocab in vocabs:
            self.adjusted_vocab.append(vocab)
        if self.coloring:
            self.coloring.extend_sequence(len(vocabs))
        self.num_labels += len(vocabs)

    @property
    def wake_word(self):
        """Construct wake word from vocab and sequence"""
        return " ".join([self.vocab[i] for i in self.sequence])

    @staticmethod
    def load_from_config(config: ContextConfig):
        """Load context instance from context config"""
        return InferenceContext(
            vocab=config.vocab,
            sequence=config.sequence,
            token_type=config.token_type,
            phone_dictionary_path=config.phone_dictionary_path,
            seed=config.seed,
        )
