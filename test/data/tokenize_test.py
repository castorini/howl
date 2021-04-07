import unittest

from howl.data.tokenize import Vocab, VocabTrie, WakeWordTokenizer


class TestVocab(unittest.TestCase):

    def test_vocab(self):
        """test Vocab class
        """
        vocab = Vocab({"Hello": 0, "World": 1}, oov_token_id=2, oov_word_repr='<OOV>')
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


class TestWakeWordTokenizer(unittest.TestCase):

    def test_decode(self):
        """test decode
        """
        vocab = Vocab({"Cat": 0, "cap": 1}, oov_token_id=2, oov_word_repr='<OOV>')
        tokenizer = WakeWordTokenizer(vocab)

        decoded_string = tokenizer.decode([0, 1, 1])
        self.assertEqual(decoded_string, "Cat cap cap")

        decoded_string = tokenizer.decode([])
        self.assertEqual(decoded_string, "")

    def test_encode(self):
        """test encode
        """
        oov_label = 2
        vocab = Vocab({"Cat": 0, "can": 1}, oov_token_id=oov_label, oov_word_repr='<OOV>')
        original_text = "Catch me if you CAN"

        # ignore oov words
        tokenizer = WakeWordTokenizer(vocab)
        encoded_output = tokenizer.encode(original_text)
        self.assertEqual(len(encoded_output), 1)
        self.assertEqual(encoded_output[0], 1)

        # consider oov words
        tokenizer = WakeWordTokenizer(vocab, False)
        encoded_output = tokenizer.encode(original_text)
        self.assertEqual(len(encoded_output), len(original_text.split()))
        self.assertEqual(encoded_output[0], oov_label)
        self.assertEqual(encoded_output[-1], 1)


if __name__ == '__main__':
    unittest.main()
