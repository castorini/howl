from torchvision.models import MobileNetV2, mobilenet_v2
import torch.nn as nn

from .base import register_model


__all__ = ['MobileNetClassifier']


@register_model('mobilenet')
class MobileNetClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = mobilenet_v2(pretrained=True)
        model = MobileNetV2(num_classes=2)
        self.model.classifier = model.classifier

    def forward(self, x, lengths):
        return self.model(x)
