import tempfile
import unittest
from pathlib import Path

from howl.workspace import Workspace


class WorkspaceTest(unittest.TestCase):
    """Test case for Workspace class"""

    def test_workspace_initialization(self):
        """Test initialization of Trainer"""

        with tempfile.TemporaryDirectory() as workspace_dir:
            workspace_path = Path(workspace_dir)
            Workspace(workspace_path, delete_existing=False)
