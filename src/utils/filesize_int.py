from os.path import getsize
from pathlib import Path


def get_filesize_bytes(filepath: Path) -> int:
    """gets file size in bytes. assumes file already exists"""
    return getsize(filename=filepath)
