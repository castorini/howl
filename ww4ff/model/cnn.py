from pydantic import BaseSettings
from torchvision.models import MobileNetV2, mobilenet_v2
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import register_model


__all__ = ['MobileNetClassifier', 'Res8']


class MobileNetSettings(BaseSettings):
    num_labels: int = 2


@register_model('mobilenet')
class MobileNetClassifier(nn.Module):
    def __init__(self, settings: MobileNetSettings = MobileNetSettings()):
        super().__init__()
        self.downsample = nn.Sequential(nn.Conv2d(1, 3, 3, padding=(1, 3)),
                                        nn.BatchNorm2d(3),
                                        nn.ReLU(),
                                        nn.MaxPool2d((1, 2)))
        self.model = mobilenet_v2(pretrained=True)
        model = MobileNetV2(num_classes=settings.num_labels)
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
    hidden_size: int = 256


@register_model('small-cnn')
class SmallCnn(nn.Module):
    def __init__(self, config: CnnSettings = CnnSettings()):
        super().__init__()
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


class Res8Settings(BaseSettings):
    num_labels: int = 2


@register_model('res8')
class Res8(nn.Module):
    def __init__(self, config: Res8Settings = Res8Settings()):
        super().__init__()
        n_maps = 45
        self.conv0 = nn.Conv2d(1, n_maps, (3, 3), padding=(1, 1), bias=False)
        self.pool = nn.AvgPool2d((3, 4))  # flipped -- better for 80 log-Mels

        self.n_layers = n_layers = 6
        self.convs = [nn.Conv2d(n_maps, n_maps, (3, 3), padding=1, bias=False) for _ in range(n_layers)]
        for i, conv in enumerate(self.convs):
            self.add_module(f'bn{i + 1}', nn.BatchNorm2d(n_maps, affine=False))
            self.add_module(f'conv{i + 1}', conv)
        self.output = nn.Linear(n_maps, config.num_labels)

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
        x = x.view(x.size(0), x.size(1), -1) # shape: (batch, feats, o3)
        x = torch.mean(x, 2)
        return self.output(x)
