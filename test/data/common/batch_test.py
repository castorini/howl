import unittest

import torch

from howl.data.common.batch import ClassificationBatch, SequenceBatch
from howl.data.transform.operator import pad, tensorize_audio_data


class TestClassificationBatch(unittest.TestCase):
    def setUp(self):
        self.audio_data = torch.zeros(16000)
        self.label = 1
        self.batch = ClassificationBatch.from_single(self.audio_data, self.label)

    def test_from_single(self):
        self.assertEqual(self.batch.audio_data.shape, torch.Size([1, len(self.audio_data)]))
        self.assertTrue(torch.equal(self.batch.audio_data, self.audio_data.unsqueeze(0)))
        self.assertTrue(self.batch.labels[0], self.label)
        self.assertTrue(self.batch.lengths.shape, [1])

    def test_to(self):

        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda"

        self.batch.to(device)

        if device == "cpu":
            self.assertEqual(self.batch.audio_data.get_device(), -1)
            self.assertEqual(self.batch.labels.get_device(), -1)
            self.assertEqual(self.batch.lengths.get_device(), -1)
        else:
            self.assertEqual(self.batch.audio_data.get_device(), 0)
            self.assertEqual(self.batch.labels.get_device(), 0)
            self.assertEqual(self.batch.lengths.get_device(), 0)


class TestSequenceBatch(unittest.TestCase):
    def setUp(self):
        self.batch_size = 3
        self.max_audio_length = 16000
        self.max_label_length = 3
        self.audio_sample_0 = torch.zeros(16000)
        self.label_0 = [0, 1, 2]
        self.audio_sample_1 = torch.zeros(12000)
        self.label_1 = [0]
        self.audio_sample_2 = torch.zeros(14000)
        self.label_2 = [0, 1]

        audio_data_list = [self.audio_sample_0, self.audio_sample_1, self.audio_sample_2]
        labels_list = [self.label_0, self.label_1, self.label_2]
        audio_data_lengths = [audio_data.size(-1) for audio_data in audio_data_list]
        labels_lengths = list(map(len, labels_list))
        audio_tensor, data = tensorize_audio_data(
            audio_data_list, labels_list=labels_list, labels_lengths=labels_lengths, input_lengths=audio_data_lengths
        )
        labels_list = torch.tensor(pad(data["labels_list"], element=-1))
        labels_lengths = torch.tensor(data["labels_lengths"])
        self.batch = SequenceBatch(audio_tensor, labels_list, torch.tensor(data["input_lengths"]), labels_lengths)

    def test_instantiation(self):
        # Note that tensorize_audio_data sort samples based on their length
        self.assertEqual(self.batch.audio_lengths[0].item(), len(self.audio_sample_0))
        self.assertEqual(self.batch.audio_lengths[1].item(), len(self.audio_sample_2))
        self.assertEqual(self.batch.audio_lengths[2].item(), len(self.audio_sample_1))
        self.assertEqual(self.batch.label_lengths[0].item(), len(self.label_0))
        self.assertEqual(self.batch.label_lengths[1].item(), len(self.label_2))
        self.assertEqual(self.batch.label_lengths[2].item(), len(self.label_1))
        self.assertEqual(self.batch.audio_data.shape, torch.Size([self.batch_size, self.max_audio_length]))
        self.assertEqual(self.batch.labels.shape, torch.Size([self.batch_size, self.max_label_length]))

    def test_to(self):

        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda"

        self.batch.to(device)

        if device == "cpu":
            self.assertEqual(self.batch.audio_data.get_device(), -1)
            self.assertEqual(self.batch.labels.get_device(), -1)
            self.assertEqual(self.batch.audio_lengths.get_device(), -1)
            self.assertEqual(self.batch.label_lengths.get_device(), -1)
        else:
            self.assertEqual(self.batch.audio_data.get_device(), 0)
            self.assertEqual(self.batch.labels.get_device(), 0)
            self.assertEqual(self.batch.audio_lengths.get_device(), 0)
            self.assertEqual(self.batch.label_lengths.get_device(), 0)


if __name__ == "__main__":
    unittest.main()
