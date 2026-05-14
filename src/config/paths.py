from pathlib import Path

from loguru import logger

from config.globals import _SHOW_LOGS


def _validate_and_log(name: str, path: Path, show_logs: bool = _SHOW_LOGS) -> Path:
    """
    Validates that a project path exists and logs its resolution at startup.

    Logs at DEBUG level when the path exists, WARNING level when it does not.
    Missing paths are not raised as errors here since some directories
    (e.g. logs/, data/) may be created lazily at runtime. Also, some deployments
    may not need specific directories, so no need to make them nor check their existence.

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


###### DIRECTORIES ######
PROJECT_DIR:            Path = _validate_and_log(name="PROJECT_DIR",          path=Path(__file__).parent.parent.parent)

# TOP LEVELS #
DATA_DIR:               Path = _validate_and_log(name="DATA_DIR",             path=PROJECT_DIR / "data")
LOGS_DIR:               Path = _validate_and_log(name="LOGS_DIR",             path=PROJECT_DIR / "logs")
SRC_DIR:                Path = _validate_and_log(name="SRC_DIR",              path=PROJECT_DIR / "src")
SCRIPTS_DIR:            Path = _validate_and_log(name="SCRIPTS_DIR",          path=PROJECT_DIR / "scripts")
TESTS_DIR:              Path = _validate_and_log(name="TESTS_DIR",            path=PROJECT_DIR / "tests")

# SOURCE CODE #
APP_DIR:                Path = _validate_and_log(name="APP_DIR",              path=SRC_DIR / "app")
CONFIG_DIR:             Path = _validate_and_log(name="CONFIG_DIR",           path=SRC_DIR / "config")
DATA_COLLECTING_DIR:    Path = _validate_and_log(name="DATA_COLLECTING_DIR",  path=SRC_DIR / "data_collecting")
MODELS_DIR:             Path = _validate_and_log(name="MODELS_DIR",           path=SRC_DIR / "models")
MONITORING_DIR:         Path = _validate_and_log(name="MONITORING_DIR",       path=SRC_DIR / "monitoring")
TRAINING_DIR:           Path = _validate_and_log(name="TRAINING_DIR",         path=SRC_DIR / "training")
UTILS_DIR:              Path = _validate_and_log(name="UTILS_DIR",            path=SRC_DIR / "utils")

# DATA #
METADATA_DATABASE_PATH: Path = _validate_and_log(name="DATABASE_PATH",        path=DATA_DIR / "data.db")
DATA_VERSIONS_DIR:      Path = _validate_and_log(name="DATA_VERSIONS_DIR",    path=DATA_DIR / "data_versions")
CORRUPT_DIR:            Path = _validate_and_log(name="CORRUPT_DIR",          path=DATA_DIR / "corrupt")
SPLITS_DIR:             Path = _validate_and_log(name="SPLITS_DIR",           path=DATA_DIR / "splits")
RAW_DIR:                Path = _validate_and_log(name="RAW_DIR",              path=DATA_DIR / "raw")

ACOUSTIC_DIR:           Path = _validate_and_log(name="ACOUSTIC_DIR",         path=RAW_DIR / "acoustic")
ELECTRIC_DIR:           Path = _validate_and_log(name="ELECTRIC_DIR",         path=RAW_DIR / "electric")
NOT_GUITAR_DIR:         Path = _validate_and_log(name="NOT_GUITAR_DIR",       path=RAW_DIR / "not_guitar")
