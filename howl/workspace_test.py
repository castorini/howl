import json
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path

import pytest
import torch

from howl.utils import test_utils
from howl.workspace import Workspace
from training.run.args import ArgumentParserBuilder, opt


class WorkspaceTest(unittest.TestCase):
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

    def test_write_args(self):
        """Test write_args"""

        with self._setup_test_env() as workspace:
            arg_key = "key"
            arg_value = "value"
            apb = ArgumentParserBuilder()
            apb.add_options(opt(f"--{arg_key}", type=str, default=arg_value))
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
            self.assertEqual(workspace.model_path(), str(workspace.path / "model.pt.bin"))

            workspace.load_model(loaded_net, best=False)
            for state_key, state_value in original_net.state_dict().items():
                self.assertTrue(torch.allclose(state_value, loaded_net.state_dict()[state_key]))

            with pytest.raises(FileNotFoundError):
                workspace.load_model(loaded_net, best=True)

            best_net = test_utils.TestNet()
            for state_key, state_value in best_net.state_dict().items():
                self.assertFalse(torch.allclose(state_value, loaded_net.state_dict()[state_key]))

            workspace.save_model(best_net, best=True)
            self.assertEqual(workspace.model_path(), str(workspace.path / "model.pt.bin"))

            workspace.load_model(loaded_net, best=False)
            for state_key, state_value in best_net.state_dict().items():
                self.assertFalse(torch.allclose(state_value, loaded_net.state_dict()[state_key]))

            workspace.load_model(loaded_net, best=True)
            for state_key, state_value in best_net.state_dict().items():
                self.assertTrue(torch.allclose(state_value, loaded_net.state_dict()[state_key]))