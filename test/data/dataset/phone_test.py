import unittest
from pathlib import Path

from howl.data.dataset import PhonePhrase, PronunciationDictionary


class TestPhonePhrase(unittest.TestCase):
    def test_basic_operations(self):
        """test PhonePhase instatniation and conversions to strgin
        """
        phone_phrase_str = " abc sil sp spn "
        pp = PhonePhrase.from_string(phone_phrase_str)
        self.assertEqual(pp.text, phone_phrase_str.strip())

    def test_idx_operations(self):
        """test idx conversions
        """
        phone_phrase_str = " abc sil sp spn def ghi sil "
        pp = PhonePhrase.from_string(phone_phrase_str)

        self.assertEqual(pp.all_idx_to_transcript_idx(0), 3)
        self.assertEqual(pp.all_idx_to_transcript_idx(1), 7)
        self.assertEqual(pp.all_idx_to_transcript_idx(2), 10)
        self.assertEqual(pp.all_idx_to_transcript_idx(3), 14)
        self.assertEqual(pp.all_idx_to_transcript_idx(4), 18)
        self.assertEqual(pp.all_idx_to_transcript_idx(5), 22)
        self.assertEqual(pp.all_idx_to_transcript_idx(6), 26)
        self.assertRaises(ValueError, pp.all_idx_to_transcript_idx, 7)

        abc_pp = PhonePhrase.from_string("abc")
        def_pp = PhonePhrase.from_string("def")
        def_ghi_pp = PhonePhrase.from_string("def ghi")
        jki_pp = PhonePhrase.from_string("jki")
        sil_pp = PhonePhrase.from_string("sil")

        self.assertEqual(pp.audible_index(abc_pp), 0)
        self.assertEqual(pp.audible_index(def_pp), 1)
        self.assertEqual(pp.audible_index(def_ghi_pp), 1)
        self.assertRaises(ValueError, pp.audible_index, jki_pp)
        self.assertRaises(ValueError, pp.audible_index, sil_pp)

        self.assertEqual(pp.audible_idx_to_all_idx(0), 0)
        self.assertEqual(pp.audible_idx_to_all_idx(1), 4)
        self.assertEqual(pp.audible_idx_to_all_idx(2), 5)
        self.assertRaises(ValueError, pp.audible_idx_to_all_idx, 3)


class TestPronounciationDictionary(unittest.TestCase):
    def test_basic_operations(self):
        """test PronounciationDictionary instatniation and phone phrase retrieval
        """
        pronounce_dict_file = Path("test/test_data/pronounciation_dictionary.txt")
        pronounce_dict = PronunciationDictionary.from_file(pronounce_dict_file)

        self.assertTrue("hey" in pronounce_dict)
        self.assertTrue("HEY" in pronounce_dict)
        self.assertTrue(" FIRE " in pronounce_dict)
        self.assertFalse(" test " in pronounce_dict)
        self.assertFalse("" in pronounce_dict)

        self.assertRaises(ValueError, pronounce_dict.encode, "")

        hey_phrases = pronounce_dict.encode("hey")
        self.assertEqual(len(hey_phrases), 1)
        self.assertEqual(hey_phrases[0].text, "hh ey1")

        fire_phrases = pronounce_dict.encode("fire")
        self.assertEqual(len(fire_phrases), 2)
        self.assertEqual(fire_phrases[0].text, "f ay1 er0")
        self.assertEqual(fire_phrases[1].text, "f ay1 r")

        fox_phrases = pronounce_dict.encode("fox")
        self.assertEqual(len(fox_phrases), 1)
        self.assertEqual(fox_phrases[0].text, "f aa1 k s")


if __name__ == "__main__":
    unittest.main()
