import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from howl.settings import KEY_TO_SETTINGS_CLASS, SETTINGS, HowlSettings
from howl.utils.dataclass import gather_dict


@dataclass
class Workspace:
    """A package that consists all the training-relevant files"""

    path: Path
    best_quality: float = float("-inf")
    delete_existing: bool = True

    def __post_init__(self):
        """Initialize Workspace by creating the directory and summary writer"""
        self.path.mkdir(parents=True, exist_ok=True)
        log_path = self.path / "logs"
        if self.delete_existing:
            shutil.rmtree(str(log_path), ignore_errors=True)
        self.summary_writer = SummaryWriter(str(log_path))

    def model_path(self, best=False):
        """Path of the trained .pt file"""
        return str(self.path / f'model{"-best" if best else ""}.pt.bin')

    def write_args(self, args):
        """Save args into cmd-args.json file

        Args:
            args (): arguments to save
        """
        with (self.path / "cmd-args.json").open("w") as file:
            json.dump(gather_dict(args), file, indent=2)

    def increment_model(self, model: nn.Module, quality):
        """Saves the model and update locally tracking performance

        Args:
            model (nn.Module): model to save
            quality (): performance of the given model
        """
        if quality > self.best_quality:
            self.save_model(model, best=True)
            self.best_quality = quality
        self.save_model(model, best=False)

    def save_model(self, model: nn.Module, best: bool = False):
        """Saves the given model (model.pt.bin or model-best.pt.bin)

        Args:
            model (nn.Module): model to save
            best (bool): if True, the given model is saved with model-best
        """
        torch.save(model.state_dict(), self.model_path(best=best))

    def load_model(self, model: nn.Module, best=True):
        """Loads saved model"""
        model.load_state_dict(torch.load(self.model_path(best=best), lambda s, l: s))

    def write_settings(self, settings: HowlSettings = SETTINGS):
        """Write settings object into JSON file"""
        with (self.path / "settings.json").open("w") as file:
            keys_to_ignore = ["_dataset", "_raw_dataset"]
            json.dump(gather_dict(settings, keys_to_ignore), file, indent=2)

    def load_settings(self, settings: HowlSettings = SETTINGS) -> HowlSettings:
        """Load settings from JSON file into provided settings object"""
        with (self.path / "settings.json").open("r") as file:
            json_settings = json.load(file)
            for key, value in json_settings.items():
                setattr(settings, key, KEY_TO_SETTINGS_CLASS[key](**value))
        return settings
