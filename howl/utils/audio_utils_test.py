import unittest

import pytest
import torch

from howl.utils import audio_utils, test_utils


class AudioUtilsTest(test_utils.HowlTest, unittest.TestCase):
    """Test case for audio utils"""

    # loaded audio data can be different based on the underlying hardware
    @pytest.mark.CI
    def test_silent_load_and_stride(self):
        """Test loading audio and striding over"""
        audio_file_path = test_utils.test_audio_file_path()
        sample_rate = 16000
        mono = True
        audio_data = audio_utils.silent_load(audio_file_path, sample_rate, mono)
        self.assertEqual(len(audio_data), 112128)
        self.assertAlmostEqual(audio_data.mean(), 2.36e-05)

        window_ms = 500
        stride_ms = 250
        # drop_incomplete = False
        windows = list(
            audio_utils.stride(torch.Tensor(audio_data), window_ms, stride_ms, sample_rate, drop_incomplete=False)
        )
        self.assertEqual(len(windows), 29)

        # drop_incomplete = True
        windows = list(
            audio_utils.stride(torch.Tensor(audio_data), window_ms, stride_ms, sample_rate, drop_incomplete=True)
        )
        self.assertEqual(len(windows), 27)
