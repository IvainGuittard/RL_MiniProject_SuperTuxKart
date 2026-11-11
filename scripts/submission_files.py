"""The files of the agent folder that are submitted on ML-Arena."""

from pathlib import Path

#: The base directory of the project, and the agent folder
ROOT = Path(__file__).resolve().parents[1]
AGENT_DIR = ROOT / "stk_actor"

#: Platform limits (see https://ml-arena.com/viewchallenge/169)
MAX_FILES = 10
MAX_SIZE = 100 * 1024 * 1024
REQUIRED = ("agent.py", "pystk_actor.py")

#: Files that are not uploaded (the shim does not need ``__init__.py``,
#: and ``learn.py`` is only used for training)
IGNORED = {"__init__.py", "__pycache__", ".DS_Store", "learn.py"}


class CheckError(Exception):
    pass


def collect_files(folder: Path) -> list[Path]:
    """Returns the files of the agent folder to upload, checking the platform
    limits (the upload is flat: no sub-folders)"""
    files, sub_folders = [], []
    for path in sorted(folder.iterdir()):
        if path.name in IGNORED or path.suffix == ".pyc":
            continue
        (sub_folders if path.is_dir() else files).append(path)

    if sub_folders:
        raise CheckError(
            "sub-folders cannot be uploaded (the upload is flat): "
            + ", ".join(path.name for path in sub_folders)
        )
    names = {path.name for path in files}
    for required in REQUIRED:
        if required not in names:
            raise CheckError(f"{required} is missing from {folder}/")
    if len(files) > MAX_FILES:
        raise CheckError(f"{len(files)} files, the maximum is {MAX_FILES}")
    size = sum(path.stat().st_size for path in files)
    if size > MAX_SIZE:
        raise CheckError(
            f"{size / 1024**2:.1f} MB in total, the maximum is "
            f"{MAX_SIZE / 1024**2:.0f} MB"
        )
    return files
