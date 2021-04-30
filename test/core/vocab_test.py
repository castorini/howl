import unittest

from howl.core.vocab import Vocab, VocabTrie


class TestVocab(unittest.TestCase):
    def test_vocab(self):
        """test Vocab class
        """
        vocab = Vocab({"Hello": 0, "World": 1}, oov_token_id=2, oov_word_repr="<OOV>")
        self.assertEqual(len(vocab), 2)

        self.assertEqual(vocab[0], "Hello")
        self.assertEqual(vocab[1], "World")
        self.assertEqual(vocab[2], "<OOV>")

        self.assertEqual(vocab["hello"], 0)
        self.assertEqual(vocab["world"], 1)
        self.assertEqual(vocab["<OOV>"], 2)
        self.assertEqual(vocab["NotInVocab"], 2)

        vocab = Vocab({"Hello": 0, "World": 1}, oov_token_id=2)
        self.assertEqual(len(vocab), 2)

        self.assertEqual(vocab[0], "Hello")
        self.assertEqual(vocab[1], "World")
        self.assertEqual(vocab[2], "[OOV]")

        self.assertEqual(vocab["hello"], 0)
        self.assertEqual(vocab["world"], 1)
        self.assertEqual(vocab["[OOV]"], 2)
        self.assertEqual(vocab["NotInVocab"], 2)

        vocab = Vocab({"Hello": 0, "World": 1})
        self.assertEqual(len(vocab), 2)

        self.assertEqual(vocab[0], "Hello")
        self.assertEqual(vocab[1], "World")
        self.assertEqual(vocab[2], "[OOV]")

        self.assertEqual(vocab["hello"], 0)
        self.assertEqual(vocab["world"], 1)
        self.assertRaises(ValueError, vocab.__getitem__, "[OOV]")
        self.assertRaises(ValueError, vocab.__getitem__, "NotInVocab")


class TestVocabTrie(unittest.TestCase):
    def test_vocab_trie(self):
        """test different cases of max splits
        """
        vocab = VocabTrie()
        vocab.add_word("Cap")
        vocab.add_word("cat")

        original_text = "Catch me if you can"

        # first cat found
        word_found, remaining_text = vocab.max_split(original_text)
        self.assertEqual(word_found, original_text[:3])
        self.assertEqual(remaining_text, original_text[3:])

        # no vocab found
        word_found, remaining_text = vocab.max_split(remaining_text)
        self.assertEqual(word_found, "")
        self.assertEqual(remaining_text, remaining_text)

        # partially matching string should not be found
        remaining_text = remaining_text[-3:]
        word_found, remaining_text = vocab.max_split(remaining_text)
        self.assertEqual(word_found, "")
        self.assertEqual(remaining_text, remaining_text)


if __name__ == "__main__":
    unittest.main()
