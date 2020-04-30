import torch
import numpy as np
from torch.utils.data import DataLoader
import torch.utils.data as tud

from ww4ff.utils.audio_preprocessor import AudioPreprocessor, FetaureExtractionType


class AudioDataLoader(tud.DataLoader):

    def __init__(self,
                 dataset: tud.Dataset,
                 audio_preprocessing: FetaureExtractionType = FetaureExtractionType.MFCC,
                 **kwargs):

        self.dataset = dataset
        self.audio_preprocessing = audio_preprocessing
        self.audio_processor = AudioPreprocessor(n_mels=303) # 3 * 101

        super().__init__(dataset=self.dataset, 
                         collate_fn=self.collate_fn, 
                         **kwargs)

    def collate_fn(self, batch):
        data = []
        targets = []
        for sample in batch:
            audio_data = np.array(sample.audio_data[:sample.sample_rate])
            if len(audio_data) < sample.sample_rate:
                audio_data = np.concatenate((audio_data, np.zeros(sample.sample_rate - len(audio_data))))

            if self.audio_preprocessing == FetaureExtractionType.MFCC:
                # TODO:: take in audio length and subsample
                audio_tensor = torch.from_numpy(self.audio_processor.compute_mfccs(audio_data).reshape(3, -1, 101)) # [3, 101, 101]
            data.append(audio_tensor)
            targets.append(sample.metadata.transcription == 'Hey Firefox')

        data = torch.stack(data, dim=0) 
        targets = torch.FloatTensor(targets)
        return data, targets
