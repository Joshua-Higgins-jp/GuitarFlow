from enum import StrEnum
from pathlib import Path
from loguru import logger
from typing import Tuple

# DIRECTORIES
PROJECT_DIR = Path(__file__).parent.parent
TESTS_DIR = PROJECT_DIR / "tests"


SRC_DIR: Path = PROJECT_DIR / "src"
APP_DIR: Path = SRC_DIR / "app"
DATA_COLLECTING_DIR: Path = SRC_DIR / "data_collecting"
UTILS_DIR: Path = SRC_DIR / "utils"
MODELS_DIR: Path = SRC_DIR / "models"

LOG_DIR: Path = Path(__file__).resolve().parent.parent / "logs"


DATA_ROOT_DIR: Path = SRC_DIR / "data"
DATABASE_PATH: Path = DATA_ROOT_DIR / "data.db"


logger.info(f"PROJECT_DIR:         {PROJECT_DIR}")

logger.info(f"TESTS_DIR:           {TESTS_DIR}")
logger.info(f"SRC_DIR:             {SRC_DIR}")
logger.info(f"APP_DIR:             {APP_DIR}")

logger.info(f"DATA_COLLECTING_DIR: {DATA_COLLECTING_DIR}")
logger.info(f"DATA_ROOT_DIR:       {DATA_ROOT_DIR}")
logger.info(f"DATABASE_PATH:       {DATABASE_PATH}")
logger.info(f"UTILS_DIR:           {UTILS_DIR}")
logger.info(f"MODELS_DIR:          {MODELS_DIR}")


class ClassLabels(StrEnum):
    """Image is labeled as either electric or acoustic."""
    ACOUSTIC = "acoustic"
    ELECTRIC = "electric"

    # NOT_GUITAR = "not_guitar"

    @classmethod
    def as_tuple(cls) -> Tuple[str, ...]:
        return tuple(member.value for member in cls)
