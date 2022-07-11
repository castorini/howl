import json
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path

import pytest
import torch

from howl.config import TrainingConfig
from howl.settings import AudioSettings, DatasetSettings, HowlSettings
from howl.utils import test_utils
from howl.utils.args_utils import ArgOption, ArgumentParserBuilder
from howl.workspace import Workspace


class WorkspaceTest(test_utils.HowlTest, unittest.TestCase):
    """Test case for Workspace class"""

    @contextmanager
    def _setup_test_env(self):
        """prepare test workspace"""
        temp_dir = tempfile.TemporaryDirectory()
        workspace = Workspace(Path(temp_dir.name), delete_existing=False)

        try:
            yield workspace
        finally:
            temp_dir.cleanup()

    def test_model_path(self):
        """Test model_path construction"""

        with self._setup_test_env() as workspace:
            self.assertEqual(workspace.model_path(), str(workspace.path / "model.pt.bin"))
            self.assertEqual(workspace.model_path(best=True), str(workspace.path / "model-best.pt.bin"))

    @pytest.mark.skip(reason="Argument Parser is being overridden by python pytest command")
    def test_write_args(self):
        """Test write_args"""

        with self._setup_test_env() as workspace:
            arg_key = "key"
            arg_value = "value"
            apb = ArgumentParserBuilder()
            apb.add_options(ArgOption(f"--{arg_key}", type=str, default=arg_value))
            args = apb.parser.parse_args()
            workspace.write_args(args)

            json_file_path = workspace.path / "cmd-args.json"
            self.assertTrue(json_file_path.exists())

            with open(json_file_path, "r") as file:
                saved_args = json.load(file)

            self.assertEqual(saved_args[arg_key], arg_value)

    def test_save_model(self):
        """Test saving and loading model"""

        with self._setup_test_env() as workspace:
            original_net = test_utils.TestNet()
            loaded_net = test_utils.TestNet()
            for state_key, state_value in original_net.state_dict().items():
                self.assertFalse(torch.allclose(state_value, loaded_net.state_dict()[state_key]))

            workspace.save_model(original_net, best=False)
            self.assertTrue(Path(workspace.model_path()).exists())

            workspace.load_model(loaded_net, best=False)
            for state_key, state_value in original_net.state_dict().items():
                self.assertTrue(torch.allclose(state_value, loaded_net.state_dict()[state_key]))

            with pytest.raises(FileNotFoundError):
                workspace.load_model(loaded_net, best=True)

            best_net = test_utils.TestNet()
            for state_key, state_value in best_net.state_dict().items():
                self.assertFalse(torch.allclose(state_value, loaded_net.state_dict()[state_key]))

            workspace.save_model(best_net, best=True)
            self.assertTrue(Path(workspace.model_path()).exists())
            self.assertTrue(Path(workspace.model_path(best=True)).exists())

            workspace.load_model(loaded_net, best=False)
            for state_key, state_value in best_net.state_dict().items():
                self.assertFalse(torch.allclose(state_value, loaded_net.state_dict()[state_key]))

            workspace.load_model(loaded_net, best=True)
            for state_key, state_value in best_net.state_dict().items():
                self.assertTrue(torch.allclose(state_value, loaded_net.state_dict()[state_key]))

    def test_increment_model(self):
        """Test incrementing and loading model"""

        with self._setup_test_env() as workspace:
            original_net = test_utils.TestNet()
            loaded_net = test_utils.TestNet()
            for state_key, state_value in original_net.state_dict().items():
                self.assertFalse(torch.allclose(state_value, loaded_net.state_dict()[state_key]))

            workspace.increment_model(original_net, quality=100)
            self.assertTrue(Path(workspace.model_path()).exists())
            self.assertTrue(Path(workspace.model_path(best=True)).exists())

            workspace.load_model(loaded_net, best=False)
            for state_key, state_value in original_net.state_dict().items():
                self.assertTrue(torch.allclose(state_value, loaded_net.state_dict()[state_key]))

            workspace.load_model(loaded_net, best=True)
            for state_key, state_value in original_net.state_dict().items():
                self.assertTrue(torch.allclose(state_value, loaded_net.state_dict()[state_key]))

            worse_net = test_utils.TestNet()
            for state_key, state_value in worse_net.state_dict().items():
                self.assertFalse(torch.allclose(state_value, loaded_net.state_dict()[state_key]))

            workspace.increment_model(worse_net, quality=50)
            self.assertTrue(Path(workspace.model_path()).exists())
            self.assertTrue(Path(workspace.model_path(best=True)).exists())

            workspace.load_model(loaded_net, best=False)
            for state_key, state_value in worse_net.state_dict().items():
                self.assertTrue(torch.allclose(state_value, loaded_net.state_dict()[state_key]))

            workspace.load_model(loaded_net, best=True)
            for state_key, state_value in worse_net.state_dict().items():
                self.assertFalse(torch.allclose(state_value, loaded_net.state_dict()[state_key]))

    def test_save_load_settings(self):
        """Test saving and loading settings"""

        with self._setup_test_env() as workspace:
            original_settings = HowlSettings()
            original_settings.audio.sample_rate = 1000
            self.assertIsInstance(original_settings.audio, AudioSettings)
            original_settings.dataset.dataset_path = "abc"
            self.assertIsInstance(original_settings.dataset, DatasetSettings)
            workspace.save_settings(original_settings)

            json_file_path = workspace.path / "settings.json"
            self.assertTrue(json_file_path.exists())

            with open(json_file_path, "r") as file:
                saved_args = json.load(file)

            self.assertIn("_audio", saved_args)
            self.assertNotIn("_dataset", saved_args)

            loaded_settings = workspace.load_settings()
            self.assertNotEqual(original_settings, loaded_settings)

            # audio setting gets saved but dataset setting doesn't get saved
            self.assertEqual(original_settings.audio, loaded_settings.audio)
            self.assertNotEqual(original_settings.dataset, loaded_settings.dataset)

    def test_save_load_config(self):
        """Test saving and loading config"""

        with self._setup_test_env() as workspace:
            training_config = TrainingConfig()
            training_config.num_epochs = 20
            workspace.save_config(training_config)

            json_file_path = workspace.path / "training_config.json"
            self.assertTrue(json_file_path.exists())

            with open(json_file_path, "r") as file:
                saved_args = json.load(file)

            self.assertIn("inference_engine_config", saved_args)
            self.assertEqual(saved_args["num_epochs"], 20)
            self.assertEqual(saved_args["workspace_path"], str(workspace.path))

            loaded_config = workspace.load_config()
            self.assertNotEqual(id(training_config), id(loaded_config))
            self.assertEqual(training_config, loaded_config)
