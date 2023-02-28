import unittest

from howl.config import TrainingConfig
from howl.trainer import Trainer
from howl.utils import test_utils


class TrainerTest(unittest.TestCase):
    """Test case for Trainer class"""

    def test_trainer_instantiation(self):
        """Test instantiation of Trainer"""
        training_config_path = test_utils.test_data_path() / "test_training_config.json"
        training_cfg = TrainingConfig.parse_file(training_config_path)
        trainer = Trainer(training_cfg)
        trainer.train(debug=True)
