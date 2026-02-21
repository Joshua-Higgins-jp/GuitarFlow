from pathlib import Path
from typing import List, Union, Optional, Iterator


def list_filepaths(
        directory: Union[str, Path],
        include_extensions: Optional[List[str]] = None,
        recursive: bool = False
) -> List[Path]:
    """
    List filepaths in a directory, optionally filtered by extensions.

    Args:
        directory: Path to the directory to list files from
        include_extensions: List of file extensions to include (e.g., ['.jpg', '.png']).
                   Extensions are case-insensitive. If None, returns all files.
        recursive: If True, search subdirectories recursively

    Returns:
        List of Path objects matching the criteria
    """
    dir_path: Path = Path(directory)

    # Choose iteration method based on recursive flag
    if recursive:
        all_items: Iterator[Path] = dir_path.rglob("*")
    else:
        all_items: Iterator[Path]  = dir_path.iterdir()

    # Filter for files only (no symlinks etc)
    real_files: List[Path] = [f for f in all_items if f.is_file()]

    # If no extensions specified, return all files
    if include_extensions is None:
        return real_files

    # Normalise extensions to lowercase and ensure they start with '.'
    normalised_extensions: List[str] = [ext.lower() if ext.startswith('.') else f'.{ext.lower()}' for ext in include_extensions]

    # Filter by extensions (case-insensitive)
    return [f for f in real_files if f.suffix.lower() in normalised_extensions]
