from howl.config import TrainingConfig
from howl.trainer import Trainer
from howl.utils import test_utils

training_config_path = test_utils.test_data_path() / "test_training_config.json"
training_cfg = TrainingConfig.parse_file(training_config_path)
trainer = Trainer(training_cfg)

# trainer._prepare_dataset(DatasetSplit.TRAINING)
trainer.train(debug=True)
