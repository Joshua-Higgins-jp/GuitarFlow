from dataclasses import dataclass, fields
from pathlib import Path
from typing import Self

from loguru import logger

from common.config import DB_NAME, DIRS_PATHS_LOG_NAME_WIDTH
from common.utils.validate_dir import validate_dir_and_log
from common.utils.validate_file import validate_fp_and_log

_PROJECT_DIR = Path(__file__).parent.parent.parent
_LOGS_DIR: Path = _PROJECT_DIR / "logs"
_LOGS_DIR.mkdir(parents=True, exist_ok=True)

# print(_PROJECT_DIR)
# print(_LOGS_DIR)


@dataclass(frozen=True)
class DirsPaths:
    """
    Immutable collection of the project's directories and key file paths.

    Instances are created only through the build classmethod, which validates
    each path (creating directories where needed) and logs the result. build
    can be called after the log sink has been configured.

    Attributes:
        PROJECT_DIR: Root directory of the project.
        LOGS_DIR: Directory holding log files.
        TESTS_DIR: Directory holding tests.
        SRC_DIR: Directory holding source code.
        APP_DIR: Directory holding the application code, inside the source directory.
        COMMON_DIR: Directory holding shared code, inside the source directory.
        DATA_MANAGEMENT_DIR: Directory holding data management code, inside the source directory.
        TRAINING_DIR: Directory holding training code, inside the source directory.
        DATA_DIR: Directory holding data files, including the database.
        IMG_RAW_DIR: Root directory of the raw image data, inside the data directory. Class subfolders
            (acoustic, electric, not guitar) are managed externally by the training logic.
        DATASETS_DIR: Directory holding the CSV files for training datasets, inside the data directory.
        QUARANTINE_DIR: Directory holding quarantine data, inside the data directory. This is data
            which is not considered system files or image data for training.
        DB_PATH: Path to the database file, inside the data directory.
        MODELS_DIR: Directory holding model files.
    """
    PROJECT_DIR: Path
    LOGS_DIR: Path
    TESTS_DIR: Path

    SRC_DIR: Path
    APP_DIR: Path
    COMMON_DIR: Path
    DATA_MANAGEMENT_DIR: Path
    TRAINING_DIR: Path

    DATA_DIR: Path
    IMG_RAW_DIR: Path
    DATASETS_DIR: Path
    QUARANTINE_DIR: Path
    DB_PATH: Path
    MODELS_DIR: Path

    @classmethod
    def build(cls) -> Self:
        """
        Validate every path, logging each one, and return a populated instance.

        Directories are validated (and created if missing) in dependency order,
        so derived paths such as APP_DIR are built from already validated
        parents. The data subdirectories and the database file are validated
        after DATA_DIR, since they live inside it.

        Returns:
            A frozen DirsPaths with every field validated.

        Raises:
            OSError: If a directory cannot be created.
            NotADirectoryError: If a directory path is still not a directory
                after creation was attempted.
            FileNotFoundError: If a required file does not exist.
        """
        project_dir: Path = validate_dir_and_log(name="PROJECT_DIR", path=_PROJECT_DIR)
        logs_dir: Path = validate_dir_and_log(name="LOGS_DIR", path=_LOGS_DIR)
        tests_dir: Path = validate_dir_and_log(name="TESTS_DIR", path=project_dir / "tests")

        src_dir: Path = validate_dir_and_log(name="SRC_DIR", path=project_dir / "src")
        app_dir: Path = validate_dir_and_log(name="APP_DIR", path=src_dir / "app")
        common_dir: Path = validate_dir_and_log(name="COMMON_DIR", path=src_dir / "common")
        data_management_dir: Path = validate_dir_and_log(name="DATA_MANAGEMENT_DIR", path=src_dir / "data_management")
        training_dir: Path = validate_dir_and_log(name="TRAINING_DIR", path=src_dir / "training")

        data_dir: Path = validate_dir_and_log(name="DATA_DIR", path=project_dir / "data")
        img_raw_dir: Path = validate_dir_and_log(name="IMG_RAW_DIR", path=data_dir / "images_raw")
        datasets_dir: Path = validate_dir_and_log(name="DATASETS_DIR", path=data_dir / "datasets")
        quarantine_dir: Path = validate_dir_and_log(name="QUARANTINE_DIR", path=data_dir / "quarantine")
        db_path: Path = validate_fp_and_log(name="DB_PATH", filepath=data_dir / DB_NAME)
        models_dir: Path = validate_dir_and_log(name="MODELS_DIR", path=project_dir / "models")

        return cls(
            PROJECT_DIR=project_dir,
            LOGS_DIR=logs_dir,
            TESTS_DIR=tests_dir,
            SRC_DIR=src_dir,
            APP_DIR=app_dir,
            COMMON_DIR=common_dir,
            DATA_MANAGEMENT_DIR=data_management_dir,
            TRAINING_DIR=training_dir,
            DATA_DIR=data_dir,
            IMG_RAW_DIR=img_raw_dir,
            DATASETS_DIR=datasets_dir,
            QUARANTINE_DIR=quarantine_dir,
            DB_PATH=db_path,
            MODELS_DIR=models_dir,
        )

    def log_paths(self) -> None:
        """
        Log every path held by this instance, one per line, with aligned columns.

        Each field is logged at INFO level as its name, padded to a fixed
        width, followed by its path. Field order matches the dataclass
        definition.
        """
        for field in fields(self):
            logger.info(f"{field.name:<{DIRS_PATHS_LOG_NAME_WIDTH}} {getattr(self, field.name)}")
