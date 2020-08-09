import logging

from .data.dataset import WordFrameLabeler, PhoneticFrameLabeler, PronunciationDictionary, PhonePhrase
from .model.inference import LabelColoring, PhoneticTranscriptSearcher, WordTranscriptSearcher
from .settings import SETTINGS


class InferenceDataContext:
    def __init__(self):
        vocab = SETTINGS.training.vocab
        self.labeler = WordFrameLabeler(vocab)
        self.coloring = None
        use_phone = SETTINGS.training.token_type == 'phone'
        if use_phone:
            self.coloring = LabelColoring()
            pd = PronunciationDictionary.from_file(SETTINGS.training.phone_dictionary)
            new_vocab = []
            for word in vocab:
                phone_phrases = pd.encode(word)
                logging.info(f'Using phonemes {str(phone_phrases)} for word {word}.')
                new_vocab.extend(x.text for x in phone_phrases)
                self.coloring.extend_sequence(len(phone_phrases))
            self.coloring.extend_sequence(1)  # negative label
            vocab = new_vocab
            phone_phrases = [PhonePhrase.from_string(x) for x in new_vocab]
            self.labeler = PhoneticFrameLabeler(phone_phrases)
        self.num_labels = len(vocab) + 1
        self.vocab = vocab
        self.negative_label = len(vocab)
        self.searcher = PhoneticTranscriptSearcher(phone_phrases, self.coloring) if use_phone else WordTranscriptSearcher(vocab)
