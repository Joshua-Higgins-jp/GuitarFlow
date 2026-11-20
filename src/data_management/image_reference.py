from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from loguru import logger

from common.constants.class_labels import ClassLabels
from common.constants.source_labels import SourceLabels


@dataclass(frozen=True)
class ImageReference:
    """
    creates a file reference object for each file found in the raw image data directory

    these are not necessarily images.. but we will need to perform some logic later
    if image

    """
    images_raw_dir: Path
    source: SourceLabels
    label: ClassLabels
    filename: str

    @property
    def filetype(self) -> str:
        return self.resolved_filepath.suffix

    @property
    def resolved_filepath(self) -> Path:
        """returns the resolved filepath for the image on disk"""
        return Path(self.images_raw_dir / f"{self.label.value}/{self.source.value}/{self.filename}")

    @classmethod
    def generate_list_of_all_files(cls, images_raw_dir: Path) -> list["ImageReference"]:
        """Build an ImageReference for every file under images_raw_dir/<label>/<source>/.

        Files are not checked for image validity. A later step quarantines anything that is not an acceptable type.

        Args:
            images_raw_dir: Root directory containing the label/source directory tree.

        Returns:
            One ImageReference per file, ordered by label, then source (enum definition order), then filename.
        """
        references: list["ImageReference"] = []

        for label in ClassLabels:
            for source in SourceLabels:
                directory = images_raw_dir / label.value / source.value
                if not directory.is_dir():
                    logger.warning(f"Expected directory is missing, skipping: {directory}")
                    continue

                for path in sorted(directory.iterdir()):
                    if not path.is_file():
                        logger.debug(f"Skipping non-file entry: {path}")
                        continue
                    references.append(cls(images_raw_dir=images_raw_dir, source=source, label=label, filename=path.name))

        logger.info(f"Found {len(references)} files under {images_raw_dir}")
        return references

    @classmethod
    def log_counts(cls, references: list["ImageReference"]) -> None:
        """Log the number of files found for each label/source combination.

        Args:
            references: References as returned by generate_list_of_all_files().
        """
        counts: Counter[tuple[ClassLabels, SourceLabels]] = Counter((ref.label, ref.source) for ref in references)

        for label in ClassLabels:
            for source in SourceLabels:
                logger.info(f"{label.value}/{source.value}: {counts[(label, source)]} files found")
        logger.info(f"Total: {len(references)} files found")
