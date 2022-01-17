import logging

import howl
from howl.config import TrainingConfig
from howl.context import InferenceContext
from howl.utils import logging_utils

# WIP; please use train.py


class Trainer:
    """Class which defines training logics"""

    def __init__(
        self, training_cfg: TrainingConfig, logger: logging.Logger = None,
    ):
        """Initialize trainer

        Args:
            training_cfg (TrainingConfig): training config that defines how to load datasets and train the model
            logger (logging.Logger): logger
        """
        self.training_cfg = training_cfg
        self.context_cfg = training_cfg.context_config
        self.context = InferenceContext.load_from_config(self.context_cfg)

        if logger is None:
            self.logger = logging_utils.setup_logger(self.__class__.__name__)

        if self.training_cfg.workspace_path is None:
            print(howl.workspaces_path() / self.context.wake_word.replace(" ", "_"))

        # def train(self):
        #     """
        #     Train the model on train datasets.
        #     """
        #     raise NotImplementedError()
        #
        # def validation(self):
        #     """
        #     Validate the model on validation datasets.
        #     """
        #     raise NotImplementedError()
