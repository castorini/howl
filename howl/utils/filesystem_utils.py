import shutil
from pathlib import Path
from typing import List, Union


def copytree(
    src_dir: Union[str, Path], dst_dir: Union[str, Path], overwrite: bool = False, ignore_patterns: List[str] = [],
):
    """copy directory from src to dst

    Args:
            src_dir: source
            dst_dir: destination
            overwrite: overwrite existing directory
            ignore_patterns: list of patterns to ignore specific files
    """
    if isinstance(src_dir, str):
        src_dir = Path(src_dir)
    if isinstance(dst_dir, str):
        dst_dir = Path(dst_dir)

    if not src_dir.exists():
        raise RuntimeError(f"The source directory does not exist: {src_dir}")

    if dst_dir.exists():
        if not overwrite:
            raise RuntimeError(
                f"The destination directory already exists: {dst_dir}. Specify overwrite=True if you want to overwrite"
            )
        else:
            # remove directory and continue copy operation
            shutil.rmtree(dst_dir)

    shutil.copytree(src_dir, dst_dir, ignore=shutil.ignore_patterns(*ignore_patterns))
