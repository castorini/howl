from dataclasses import dataclass
import math


@dataclass
class ConfusionMatrix:
    tp: int = 0
    fp: int = 0
    tn: int = 0
    fn: int = 0

    def increment(self, pred: bool, label: bool):
        if pred and label:
            self.tp += 1
        elif pred and not label:
            self.fp += 1
        elif not pred and label:
            self.fn += 1
        elif not pred and not label:
            self.tn += 1

    @property
    def mcc(self) -> float:
        tp, tn, fp, fn = self.tp, self.tn, self.fp, self.fn
        num = tp * tn - fp * fn
        denom = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        if denom == 0:
            denom = 1
        return num / denom
