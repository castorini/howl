import unittest

from howl.data.common.vocab import Vocab
from howl.data.tokenizer import WakeWordTokenizer


class TestWakeWordTokenizer(unittest.TestCase):
    def test_decode(self):
        """test decode
        """
        vocab = Vocab({"Cat": 0, "cap": 1}, oov_token_id=2, oov_word_repr="<OOV>")
        tokenizer = WakeWordTokenizer(vocab)

        decoded_string = tokenizer.decode([0, 1, 1])
        self.assertEqual(decoded_string, "Cat cap cap")

        decoded_string = tokenizer.decode([])
        self.assertEqual(decoded_string, "")

    def test_encode(self):
        """test encode
        """
        oov_label = 2
        vocab = Vocab({"Cat": 0, "can": 1}, oov_token_id=oov_label, oov_word_repr="<OOV>")
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


if __name__ == "__main__":
    unittest.main()
