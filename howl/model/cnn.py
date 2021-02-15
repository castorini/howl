from typing import Tuple

from pydantic import BaseSettings
from torchvision.models import MobileNetV2, mobilenet_v2
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import RegisteredModel


__all__ = ['MobileNetClassifier', 'Res8']


class MobileNetClassifier(RegisteredModel, name='mobilenet'):
    def __init__(self, num_labels: int):
        super().__init__(num_labels)
        self.downsample = nn.Sequential(nn.Conv2d(1, 3, 3, padding=(1, 3)),
                                        nn.BatchNorm2d(3),
                                        nn.ReLU(),
                                        nn.MaxPool2d((1, 2)))
        self.model = mobilenet_v2(pretrained=True)
        model = MobileNetV2(num_classes=num_labels)
        self.model.classifier = model.classifier

    def forward(self, x, lengths):
        x = x[:, :1]  # log-Mels only
        x = self.downsample(x)
        return self.model(x)


class CnnSettings(BaseSettings):
    num_labels: int = 2
    num_maps1: int = 48
    num_maps2: int = 64
    num_hidden_input: int = 384
    hidden_size: int = 128


class SmallCnn(RegisteredModel, name='small-cnn'):
    def __init__(self, num_labels: int, config: CnnSettings = CnnSettings()):
        super().__init__(num_labels)
        self.config = config
        conv0 = nn.Conv2d(1, config.num_maps1, (8, 16), padding=(4, 0), stride=(2, 2), bias=True)
        pool = nn.MaxPool2d(2)
        conv1 = nn.Conv2d(config.num_maps1, config.num_maps2, (5, 5), padding=2, stride=(2, 1), bias=True)
        self.encoder1 = nn.Sequential(conv0,
                                      nn.ReLU(),
                                      pool,
                                      nn.BatchNorm2d(config.num_maps1, affine=True))
        self.encoder2 = nn.Sequential(conv1,
                                      nn.ReLU(),
                                      pool,
                                      nn.BatchNorm2d(config.num_maps2, affine=True))
        self.output = nn.Sequential(nn.Linear(config.num_hidden_input, config.hidden_size),
                                    nn.ReLU(),
                                    nn.Dropout(0.1),
                                    nn.Linear(config.hidden_size, config.num_labels))

    def forward(self, x, lengths):
        x = x[:, :1]  # log-Mels only
        x = x.permute(0, 1, 3, 2)  # (time, frequency)
        x1 = self.encoder1(x)
        x2 = self.encoder2(x1)
        x2 = x2.view(x2.size(0), x2.size(1), -1)
        x = x2.view(x2.size(0), -1)
        return self.output(x)


class SequentialCnn(RegisteredModel, name='seq-cnn'):
    def __init__(self, num_labels: int, config: CnnSettings = CnnSettings()):
        super().__init__(num_labels)
        self.config = config
        conv0 = nn.Conv2d(1, config.num_maps1, (20, 16), padding=(10, 0), stride=(1, 2), bias=True)
        pool = nn.MaxPool2d(2)
        conv1 = nn.Conv2d(config.num_maps1, config.num_maps2, (5, 5), padding=2, stride=(2, 1), bias=True)
        self.encoder1 = nn.Sequential(conv0,
                                      nn.ReLU(),
                                      pool,
                                      nn.BatchNorm2d(config.num_maps1, affine=True))
        self.encoder2 = nn.Sequential(conv1,
                                      nn.ReLU(),
                                      pool,
                                      nn.BatchNorm2d(config.num_maps2, affine=True))
        self.output = nn.Sequential(nn.Linear(3 * config.num_maps2, config.hidden_size),
                                    nn.ReLU(),
                                    nn.Dropout(0.1),
                                    nn.Linear(config.hidden_size, config.num_labels))

    def compute_length(self, length: int):
        length = int((length + 2 * 10 - (20 - 1) - 1) / 1 + 1)
        length //= 2
        length = int((length + 2 * 2 - (5 - 1) - 1) / 2 + 1)
        length //= 2
        return length

    def forward(self, x, lengths):
        x = x[:, :1]  # log-Mels only
        x = x.permute(0, 1, 3, 2)  # (time, frequency)
        x1 = self.encoder1(x)
        x2 = self.encoder2(x1)
        x = x2.permute(2, 0, 1, 3).contiguous()
        x = x.view(x.size(0), x.size(1), x.size(2) * x.size(3))
        return self.output(x)


class Res8Settings(BaseSettings):
    num_labels: int = 2
    pooling: Tuple[int, int] = (3, 4)
    num_maps: int = 45


class Res8(RegisteredModel, name='res8'):
    def __init__(self, num_labels: int, config: Res8Settings = Res8Settings()):
        super().__init__(num_labels)
        n_maps = config.num_maps
        self.conv0 = nn.Conv2d(1, n_maps, (3, 3), padding=(1, 1), bias=False)
        self.pool = nn.AvgPool2d(config.pooling)  # flipped -- better for 80 log-Mels

        self.n_layers = n_layers = 6
        self.convs = [nn.Conv2d(n_maps, n_maps, (3, 3), padding=1, bias=False) for _ in range(n_layers)]
        for i, conv in enumerate(self.convs):
            self.add_module(f'bn{i + 1}', nn.BatchNorm2d(n_maps, affine=False))
            self.add_module(f'conv{i + 1}', conv)
        self.output = nn.Linear(n_maps, num_labels)

    def forward(self, x, lengths):
        x = x[:, :1]  # log-Mels only
        x = x.permute(0, 1, 3, 2).contiguous()  # Original res8 uses (time, frequency) format
        for i in range(self.n_layers + 1):
            y = F.relu(getattr(self, f'conv{i}')(x))
            if i == 0:
                if hasattr(self, 'pool'):
                    y = self.pool(y)
                old_x = y
            if i > 0 and i % 2 == 0:
                x = y + old_x
                old_x = x
            else:
                x = y
            if i > 0:
                x = getattr(self, f'bn{i}')(x)
        x = x.view(x.size(0), x.size(1), -1)  # shape: (batch, feats, o3)
        x = torch.mean(x, 2)
        return self.output(x)
