import unittest

from howl.data.tokenize import Vocab
from howl.data.searcher import WordTranscriptSearcher
from howl.settings import SETTINGS


class TestWordTranscriptSearcher(unittest.TestCase):

    def test_search(self):
        """test search
        """
        SETTINGS.inference_engine.inference_sequence = [0, 1]
        vocab = Vocab({"hello": 0, "world": 1}, oov_token_id=2, oov_word_repr='<OOV>')
        tokenizer = WordTranscriptSearcher(vocab)

        self.assertTrue(tokenizer.search("hello world"))
        self.assertTrue(tokenizer.search("hello world and bye"))
        self.assertFalse(tokenizer.search("completely incorrect"))
        self.assertTrue(tokenizer.search("Hello World"))

    def test_contains_any(self):
        """test contains_any
        """
        SETTINGS.inference_engine.inference_sequence = [0, 1]
        vocab = Vocab({"hello": 0, "world": 1}, oov_token_id=2, oov_word_repr='<OOV>')
        tokenizer = WordTranscriptSearcher(vocab)

        self.assertTrue(tokenizer.contains_any("hello world"))
        self.assertTrue(tokenizer.contains_any("hello world and bye"))
        self.assertFalse(tokenizer.contains_any("completely incorrect"))
        self.assertTrue(tokenizer.contains_any("abc Hello"))

    def test_count_vocab(self):
        """test count_vocab
        """
        SETTINGS.inference_engine.inference_sequence = [0, 1]
        vocab = Vocab({"Hello": 0, "World": 1}, oov_token_id=2, oov_word_repr='<OOV>')
        tokenizer = WordTranscriptSearcher(vocab)

        # test string with valid vocabs
        counter = tokenizer.count_vocab("hello world and so on hello")
        self.assertEqual(len(counter), len(vocab))
        self.assertEqual(counter["Hello"], 2)
        self.assertEqual(counter["World"], 1)

        # test string with invalid vocabs
        counter = tokenizer.count_vocab("test test test test")
        self.assertEqual(len(counter), len(vocab))
        self.assertEqual(counter["Hello"], 0)
        self.assertEqual(counter["World"], 0)

        # test empty string
        counter = tokenizer.count_vocab("")
        self.assertEqual(len(counter), len(vocab))
        self.assertEqual(counter["Hello"], 0)
        self.assertEqual(counter["World"], 0)


if __name__ == '__main__':
    unittest.main()
