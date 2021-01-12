import multiprocessing as mp

import torch.utils.data as tud

from howl.data.dataset import AudioDataset, DatasetType


class StandardAudioDataLoaderBuilder:
    def __init__(self,
                 dataset: AudioDataset,
                 num_workers=mp.cpu_count(),
                 collate_fn=None):
        self.dataset = dataset
        self.num_workers = num_workers
        self.collate_fn = collate_fn

    def build(self, batch_size: int) -> tud.DataLoader:
        if self.dataset.set_type == DatasetType.TRAINING:
            return tud.DataLoader(self.dataset,
                                  batch_size=batch_size,
                                  drop_last=True,
                                  shuffle=True,
                                  num_workers=self.num_workers,
                                  collate_fn=self.collate_fn)
        else:
            return tud.DataLoader(self.dataset,
                                  batch_size=batch_size,
                                  num_workers=self.num_workers,
                                  collate_fn=self.collate_fn)
