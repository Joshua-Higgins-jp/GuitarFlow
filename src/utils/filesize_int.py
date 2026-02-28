from os.path import getsize
from pathlib import Path


def get_filesize_bytes(filepath: Path) -> int:
    """
    Gets file size in bytes. Assumes file already exists.
    """
    return getsize(filename=filepath)
