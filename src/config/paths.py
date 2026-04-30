from pathlib import Path
from loguru import logger


SHOW_LOGS = False

def _validate_and_log(name: str, path: Path, show_logs: bool = SHOW_LOGS) -> Path:
    """
    Validates that a project path exists and logs its resolution at startup.

    Logs at DEBUG level when the path exists, WARNING level when it does not.
    Missing paths are not raised as errors here since some directories
    (e.g. logs/, data/) may be created lazily at runtime.

    Parameters:
        name: Name of the project path.
        path: Path to the project path.
        show_logs: Whether to log path names. warning or higher always logs.

    Returns:
        Path to the project path; same as input path.
    """
    if path.exists():
        if show_logs:
            logger.info(f"{name:<24} {path}")
    else:
        logger.warning(f"{name:<24} {path}  ← does not exist")
    return path


PROJECT_DIR = _validate_and_log(name="PROJECT_DIR", path=Path(__file__).parent.parent.parent)
TESTS_DIR:   Path = _validate_and_log(name="TESTS_DIR", path=PROJECT_DIR / "tests")
DATA_DIR:    Path = _validate_and_log(name="DATA_DIR", path=PROJECT_DIR / "data")
DATABASE_PATH: Path = _validate_and_log(name="DATABASE_PATH", path=DATA_DIR / "data.db")
SRC_DIR:     Path = _validate_and_log(name="SRC_DIR", path=PROJECT_DIR / "src")
APP_DIR:     Path = _validate_and_log(name="APP_DIR", path=SRC_DIR / "app")
CONFIG_DIR:  Path = _validate_and_log(name="CONFIG_DIR", path=SRC_DIR / "config")
DATA_COLLECTING_DIR: Path = _validate_and_log(name="DATA_COLLECTING_DIR", path=SRC_DIR / "data_collecting")
MODELS_DIR:  Path = _validate_and_log(name="MODELS_DIR", path=SRC_DIR / "models")
MONITORING_DIR: Path = _validate_and_log(name="MONITORING_DIR", path=SRC_DIR / "monitoring")
TRAINING_DIR: Path = _validate_and_log(name="TRAINING_DIR", path=SRC_DIR / "training")
UTILS_DIR:   Path = _validate_and_log(name="UTILS_DIR", path=SRC_DIR / "utils")
