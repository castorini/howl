from dataclasses import dataclass
from pathlib import Path
import json
import shutil

from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn as nn

from .dataclass import gather_dict


@dataclass
class Workspace(object):
    path: Path
    best_quality: float = -10000.0
    delete_existing: bool = True

    def __post_init__(self):
        self.path.mkdir(parents=True, exist_ok=True)
        log_path = self.path / 'logs'
        if self.delete_existing:
            shutil.rmtree(str(log_path), ignore_errors=True)
        self.summary_writer = SummaryWriter(str(log_path))

    def model_path(self, best=False):
        return str(self.path / f'model{"-best" if best else ""}.pt.bin')

    def write_setting(self, setting):
        with open(self.path / 'config.json', 'w') as f:
            json.dump(gather_dict(config), f, indent=2)

    def write_args(self, args):
        with open(self.path / 'cmd-args.json', 'w') as f:
            json.dump(gather_dict(args), f, indent=2)

    def increment_model(self, model, quality):
        if quality > self.best_quality:
            self.save_model(model, best=True)
            self.best_quality = quality
        self.save_model(model, best=False)

    def save_model(self, model: nn.Module, best=False):
        torch.save(model.state_dict(), self.model_path(best=best))

    def load_model(self, model: nn.Module, best=True):
        model.load_state_dict(torch.load(self.model_path(best=best), lambda s, l: s))