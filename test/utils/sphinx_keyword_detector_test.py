import unittest

from howl.utils.sphinx_keyword_detector import SphinxKeywordDetector


class TestSphinxKeywordDetector(unittest.TestCase):

    def test_detect(self):
        """test word detection from an audio file
        """

        hello_world_file = "test/test_data/sphinx_keyword_detector/hello_world.wav"
        hello_extractor = SphinxKeywordDetector("hello")
        self.assertTrue(len(hello_extractor.detect(hello_world_file)) > 0)
        world_extractor = SphinxKeywordDetector("world")
        self.assertTrue(len(world_extractor.detect(hello_world_file)) > 0)

        hey_fire_fox_file = "test/test_data/sphinx_keyword_detector/hey_fire_fox.wav"
        hey_extractor = SphinxKeywordDetector("hey")
        self.assertTrue(len(hey_extractor.detect(hey_fire_fox_file)) > 0)
        fire_extractor = SphinxKeywordDetector("fire")
        self.assertTrue(len(fire_extractor.detect(hey_fire_fox_file)) > 0)
        fox_extractor = SphinxKeywordDetector("fox")
        self.assertTrue(len(fox_extractor.detect(hey_fire_fox_file)) > 0)


if __name__ == '__main__':
    unittest.main()
