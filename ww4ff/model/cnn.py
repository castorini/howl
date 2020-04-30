from dataclasses import dataclass

from torchvision.models import MobileNetV2, mobilenet_v2
import torch.nn as nn


@dataclass
class MNClassifierConfig(object):
    num_labels: int


class MobileNetClassifier(nn.Module):

    def __init__(self, config: MNClassifierConfig):
        super().__init__()
        self.model = mobilenet_v2(pretrained=True)
        model = MobileNetV2(num_classes=config.num_labels)
        self.model.classifier = model.classifier

    def forward(self, x):
        return self.model(x)


def convert_half(module: nn.Module):
    for mod in module.modules():
        if not isinstance(mod, nn.BatchNorm2d):
            mod.half()