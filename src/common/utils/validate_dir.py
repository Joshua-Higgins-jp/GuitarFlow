from pathlib import Path

from loguru import logger
from common.config import DIRS_PATHS_LOG_NAME_WIDTH


def validate_dir_and_log(name: str, path: Path) -> Path:
    """
    Ensure a project directory exists, creating it if necessary.

    Logs at WARNING level when the directory is missing and creation is
    attempted, and at ERROR level if creation fails. Nothing is logged when
    the directory already exists. Creation is attempted once only; there is no
    retry. If the directory still cannot be confirmed afterwards, the error is
    raised to the caller.

    Parameters:
        name: Name of the project path, used in log and error messages.
        path: Path to the project directory.

    Returns:
        Path to the project directory; same as input path.

    Raises:
        OSError: If the directory cannot be created (e.g. permissions, read-only
            filesystem, or the path exists as a file).
        NotADirectoryError: If the path is still not a directory after creation.
    """
    _LEN: int = DIRS_PATHS_LOG_NAME_WIDTH

    if path.is_dir():
        return path

    logger.warning(f"{name:<{_LEN}} {path}  ← does not exist, attempting to create")

    try:
        path.mkdir(parents=True, exist_ok=True)

    except OSError as oss_err:
        logger.error(f"{name:<{_LEN}} {path}  ← could not be created, cause: {oss_err}")
        raise

    if not path.is_dir():
        raise NotADirectoryError(f"{name:<{_LEN}} ({path}) is still not a directory after mkdir")

    return path
