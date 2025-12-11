from enum import Enum
from pathlib import Path

from loguru import logger

from src.data_collecting.collectors.collector_models import ImageLabel, ImageSource

DATA_ROOT_DIR: Path = Path(__file__).resolve().parent.parent.parent / "data"
# print(f"actual image data root directory is:\n---> `{DATA_ROOT_DIR}`")


class DatasetStage(Enum):
    """Stages in the ML data pipeline."""
    RAW = "raw"
    CURATED = "curated"
    SPLITS = "splits"


class SplitType(Enum):
    """Train/validation/test split types."""
    TRAIN = "train"
    VAL = "val"
    TEST = "test"


class DataDirectoryManager:
    """
    Manages directory structure for ML image collection and provides path access.

    Directory Structure:
        data_root/
        ├── raw/
        │   ├── pixabay/
        │   │   ├── electric/
        │   │   └── acoustic/
        │   ├── unsplash/
        │   │   ├── electric/
        │   │   └── acoustic/
        │   └── pexels/
        │       ├── electric/
        │       └── acoustic/
        ├── curated/
        │   ├── electric/
        │   └── acoustic/
        └── splits/
            ├── train/
            │   ├── electric/
            │   └── acoustic/
            ├── val/
            │   ├── electric/
            │   └── acoustic/
            └── test/
                ├── electric/
                └── acoustic/

    Attributes:
        class_log_pfx: Logging prefix for all class operations
        data_root: Root directory for all data storage
    """
    def __init__(
            self,
            init_data_dirs: bool = True,
            data_root: Path = DATA_ROOT_DIR
    ) -> None:
        """
        Initialize the DataDirectoryManager.

        Args:
            init_data_dirs: If True, automatically creates all directory structures.
            data_root: Root directory path for data storage.
        """
        self.class_log_pfx: str = "DATA DIRECTORY MANAGER"
        self.data_root: Path = data_root

        if init_data_dirs:
            self._init_all_data_dirs()

    def _init_all_data_dirs(self) -> None:
        """Initialize complete directory structure for all pipeline stages."""
        log_pfx: str = "INIT ALL DATA DIRS"
        logger.info(f"{self.class_log_pfx}: {log_pfx}: Starting directory initialization")

        # Create raw directories for each source
        for source in ImageSource:
            self._init_raw_dirs(source)

        # Create curated directories (source-agnostic)
        self._init_curated_dirs()

        # Create train/val/test split directories
        self._init_split_dirs()

        logger.info(f"{self.class_log_pfx}: {log_pfx}: Directory initialization complete")

    def _init_raw_dirs(self, image_source: ImageSource) -> None:
        """
        Create raw data directory structure for a specific image source.

        Args:
            image_source: The ImageSource enum value (e.g., PIXABAY)
        """
        log_pfx: str = "INIT RAW DIRS"
        service_name: str = image_source.value.lower()

        for label in ImageLabel:
            label_name: str = label.value.lower()
            raw_dir: Path = self.data_root / DatasetStage.RAW.value / service_name / label_name
            raw_dir.mkdir(parents=True, exist_ok=True)

            logger.info(
                f"{self.class_log_pfx}: {log_pfx}: "
                f"Ensured directory exists: {raw_dir}"
            )

    def _init_curated_dirs(self) -> None:
        """Create curated data directory structure (source-agnostic)."""
        log_pfx: str = "INIT CURATED DIRS"

        for label in ImageLabel:
            label_name: str = label.value.lower()
            curated_dir: Path = self.data_root / DatasetStage.CURATED.value / label_name
            curated_dir.mkdir(parents=True, exist_ok=True)

            logger.info(
                f"{self.class_log_pfx}: {log_pfx}: "
                f"Ensured directory exists: {curated_dir}"
            )

    def _init_split_dirs(self) -> None:
        """Create train/val/test split directory structure."""
        log_pfx: str = "INIT SPLIT DIRS"

        for split in SplitType:
            for label in ImageLabel:
                label_name: str = label.value.lower()
                split_dir: Path = (
                        self.data_root / DatasetStage.SPLITS.value / split.value / label_name
                )
                split_dir.mkdir(parents=True, exist_ok=True)

                logger.info(
                    f"{self.class_log_pfx}: {log_pfx}: "
                    f"Ensured directory exists: {split_dir}"
                )

    # Directory path getters

    def get_raw_dir(
            self,
            image_source: ImageSource,
            label: ImageLabel
    ) -> Path:
        """
        Get path to raw data directory for specific source and label.

        Args:
            image_source: The image source service
            label: The image label/category

        Returns:
            Path object to the raw directory
        """
        return (
                self.data_root / DatasetStage.RAW.value /
                image_source.value.lower() / label.value.lower()
        )

    def get_curated_dir(self, label: ImageLabel) -> Path:
        """
        Get path to curated data directory for specific label.

        Args:
            label: The image label/category

        Returns:
            Path object to the curated directory
        """
        return self.data_root / DatasetStage.CURATED.value / label.value.lower()

    def get_split_dir(
            self,
            split: SplitType,
            label: ImageLabel
    ) -> Path:
        """
        Get path to train/val/test split directory for specific label.

        Args:
            split: The split type (train/val/test)
            label: The image label/category

        Returns:
            Path object to the split directory
        """
        return (
                self.data_root / DatasetStage.SPLITS.value /
                split.value / label.value.lower()
        )

    # Image file path getters

    def get_raw_image(
            self,
            image_source: ImageSource,
            label: ImageLabel,
            filename: str
    ) -> Path:
        """
        Get path to a specific image in raw directory.

        Args:
            image_source: The image source service
            label: The image label/category
            filename: Name of the image file

        Returns:
            Path object to the image file
        """
        return self.get_raw_dir(image_source, label) / filename

    def get_curated_image(
            self,
            label: ImageLabel,
            filename: str
    ) -> Path:
        """
        Get path to a specific image in curated directory.

        Args:
            label: The image label/category
            filename: Name of the image file

        Returns:
            Path object to the image file
        """
        return self.get_curated_dir(label) / filename

    def get_split_image(
            self,
            split: SplitType,
            label: ImageLabel,
            filename: str
    ) -> Path:
        """
        Get path to a specific image in a split directory.

        Args:
            split: The split type (train/val/test)
            label: The image label/category
            filename: Name of the image file

        Returns:
            Path object to the image file
        """
        return self.get_split_dir(split, label) / filename
