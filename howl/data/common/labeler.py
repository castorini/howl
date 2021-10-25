import string
from typing import List

from howl.data.common.frame import FrameLabelData
from howl.data.common.metadata import AudioClipMetadata
from howl.data.common.phone import PhoneEnum, PhonePhrase, PronunciationDictionary
from howl.data.common.vocab import Vocab

__all__ = ["FrameLabeler", "WordFrameLabeler", "PhoneticFrameLabeler"]


class FrameLabeler:
    def compute_frame_labels(self, metadata: AudioClipMetadata) -> FrameLabelData:
        raise NotImplementedError


class PhoneticFrameLabeler(FrameLabeler):
    def __init__(self, pronounce_dict: PronunciationDictionary, phrases: List[PhonePhrase]):
        self.pronounce_dict = pronounce_dict
        self.phrases = phrases
        punctuation_to_replace = str.maketrans(
            {"‘": "'", "’": "'", "”": '"', "“": '"', "—": "-", "ä": "a", "ö": "o", "ō": "o", "é": "e", "à": "a"}
        )
        punctuation_to_remove = str.maketrans({key: None for key in string.punctuation})
        # First transformation is None because we want to process the original word first
        self.punctuation_transforms = [
            None,
            punctuation_to_replace,
            punctuation_to_remove,
        ]

    def transform(self, original_word: str) -> PhonePhrase:
        """Transform the word into list of Phones

        TODO:
        the function attempts to find list of the longest phone phrase sequences
        However, this is not ideal for some edge cases.
        for example, let's say helloworld might be broken into [hellow, or, ld]
        while [hello, world] would be the most appropriate breakdown

        Args:
            original_word (str): word to transform

        Raises:
            ValueError: if the word cannot be broken down to phonemes

        Returns:
            PhonePhrase: phone phrase for the given word
        """
        phrases = PhonePhrase([])
        word = original_word
        idx = len(word)
        while idx > 0:
            while word[:idx] not in self.pronounce_dict and idx > 0:
                idx -= 1
            phrase = None
            try:
                # TODO:: we currently use single representation for simplicity
                phrase = self.pronounce_dict.encode(word[:idx])[0]
                phrases.extend(phrase)
            except ValueError:
                if word == "<unk>":
                    phrase = PhonePhrase.from_string(PhoneEnum.SPEECH_UNKNOWN.value)
                    idx = -1
                    phrases.extend(phrase)
                    continue
                else:
                    raise ValueError(
                        f"word {word} ({original_word}) does not have corresponding phoneme representation"
                    )

            word = word[idx:]
            idx = len(word)

        return phrases

    def compute_frame_labels(self, metadata: AudioClipMetadata) -> FrameLabelData:
        """Process metadata to compute labels FrameLabelData which will be used for training

        Args:
            metadata (AudioClipMetadata): information about the sample

        Returns:
            FrameLabelData: labels for the current sample with respect to frame
        """
        frame_labels = dict()
        char_indices = []
        start_timestamp = []

        phonetic_transcription = PhonePhrase([])

        # breaking down each word into PhonePhrase
        for original_word in metadata.transcription.split():
            # if transformation fails, it might be due to some invalid characters
            # in this case, we change the character to the other appropriate character
            phrase = None
            for punctuation_transform in self.punctuation_transforms:
                if punctuation_transform is not None:
                    original_word = original_word.translate(punctuation_transform)
                    if len(original_word) == 0:
                        break
                try:
                    phrase = self.transform(original_word)
                    break
                except ValueError:
                    pass

            if phrase:
                phonetic_transcription.extend(phrase)
            elif len(original_word) > 0:
                print(f"Failed to find phonemes for {original_word} {[ord(c) for c in original_word]}")

        # TODO: idx might not be the correct label due to repeated phonemes
        for idx, phrase in enumerate(self.phrases):
            start = 0
            while True:
                try:
                    start = phonetic_transcription.audible_index(phrase, start)
                except ValueError:
                    break

                # TDOO: metadata.end_timestamps contains when the given character in the transcript finishes
                #       in order to label the the phonemes right, we need a mapping from characters to phonemes
                # # compute where the phrase located in the transcript
                # start = phonetic_transcription.all_idx_to_transcript_idx(
                #     phonetic_transcription.audible_idx_to_all_idx(start)
                # )

                # phonetic_transcription = list of phones for the transcription
                # end_timestamps = when character of the given index in the transcription is pronounced
                # phoeneme_mapping = from character index to phone index in phonetic_transcription

                # char_index = phoneme_mapping[start]
                # end_time = metadata.end_timestamps[char_index]
                # frame_labels[end_time] = idx
                frame_labels[metadata.end_timestamps[start]] = idx
                start += 1

        # TODO: process phonetic_transcription to compute valid FrameLabelData
        return FrameLabelData(frame_labels, start_timestamp, char_indices)


class WordFrameLabeler(FrameLabeler):
    def __init__(self, vocab: Vocab):
        self.vocab = vocab

    def compute_frame_labels(self, metadata: AudioClipMetadata) -> FrameLabelData:
        frame_labels = dict()
        start_timestamp = []
        char_indices = []

        char_idx = 0
        for word in metadata.transcription.split():
            vocab_found, remaining_transcript = self.vocab.trie.max_split(word)
            word_size = len(word.rstrip())

            # if the current word is in vocab, store necessary informations
            if vocab_found and remaining_transcript == "":
                label = self.vocab[word]
                end_timestamp = metadata.end_timestamps[char_idx + word_size - 1]
                frame_labels[end_timestamp] = label
                char_indices.append((label, list(range(char_idx, char_idx + word_size))))
                start_timestamp.append((label, metadata.end_timestamps[char_idx - 1] if char_idx > 0 else 0.0,))

            char_idx += word_size + 1  # space

        return FrameLabelData(frame_labels, start_timestamp, char_indices)
