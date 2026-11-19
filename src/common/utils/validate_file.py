from pathlib import Path

from loguru import logger
from common.config import DIRS_PATHS_LOG_NAME_WIDTH


def validate_fp_and_log(name: str, filepath: Path) -> Path:
    """
    Check that a filepath points to an existing regular file and return it resolved.

    Symlinks are permitted. The filepath is fully resolved (following any chain
    of symlinks), and the final target must exist and be a regular file. A
    broken symlink, a symlink loop, a missing filepath, or a filepath that
    resolves to a directory (or other non-file) all raise FileNotFoundError.

    Args:
        name: A short human-readable label for the file (for example
            "config_file"), used to identify it in error messages.
        filepath: The filepath to validate. Must be a pathlib.Path, and may be
            relative, absolute, or contain symlinks.

    Returns:
        The fully resolved absolute filepath of the existing file, with all
        symlinks followed.

    Raises:
        FileNotFoundError: If the filepath does not exist, is a broken symlink,
            is part of a symlink loop, or does not resolve to a regular file.
    """
    _LEN: int = DIRS_PATHS_LOG_NAME_WIDTH

    try:
        # strict=True raises if any component (including a symlink target)
        # does not exist.
        resolved_filepath: Path = filepath.resolve(strict=True)

    except FileNotFoundError as exc:
        logger.error(f"{name:<{_LEN}} {filepath}  ← does not exist or is a broken symlink")
        raise FileNotFoundError(
            f"{name} ({filepath}) not found (or broken symlink)"
        ) from exc

    except (RuntimeError, OSError) as exc:
        # Symlink loops raise RuntimeError (Python < 3.13) or OSError (3.13+).
        logger.error(f"{name:<{_LEN}} {filepath}  ← could not be resolved ({exc})")
        raise FileNotFoundError(
            f"{name} ({filepath}) could not be resolved (possible symlink loop)"
        ) from exc

    if not resolved_filepath.is_file():
        logger.error(f"{name:<{_LEN}} {resolved_filepath}  ← is not a regular file")
        raise FileNotFoundError(
            f"{name} ({resolved_filepath}) does not resolve to a regular file"
        )

    return resolved_filepath
