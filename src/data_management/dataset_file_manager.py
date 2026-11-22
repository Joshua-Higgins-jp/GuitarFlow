from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Iterator, Optional

from loguru import logger

from common.constants.accepted_image_formats import AcceptedImageFormats
from common.constants.class_labels import ClassLabels
from common.constants.source_labels import SourceLabels
from common.utils.rename_file_with_lowercase_extension import rename_file_with_lowercase_extension
from data_management.image_reference import ImageDatasetFileReference


class QuarantineStage(StrEnum):
    FILESYSTEM__DIR_STRUCTURE_CHECK = "filesystem__dir_structure_check"
    FILESYSTEM__FILETYPES_CHECK = "filesystem__filetypes_check"
    FILESYSTEM__DUPLICATES_CHECK = "filesystem__duplicates_check"

    IMAGE = "image"
    DUPLICATE = "duplicate"


@dataclass(frozen=True)
class UnexpectedDirEvent:
    directory: Path
    reason: str


@dataclass(frozen=True)
class ValidateDirStructureResult:
    results: list[UnexpectedDirEvent]

    @property
    def has_problems(self) -> bool:
        return self.num_problems > 0

    @property
    def num_problems(self) -> int:
        return len(self.results)

    def log_results(self, relative_to_dir: Path) -> None:
        for result in self.results:
            logger.warning(f"Directory `{result.directory.relative_to(relative_to_dir)}` is {result.reason}")

    def runtime_error_str(self) -> str:
        return (
            f"{self.num_problems} unexpected directory entries found. "
            f"Check logs and rectify, then try again."
        )


@dataclass(frozen=True)
class QuarantineEntry:
    """
    One flagged file.

    Attributes:
        file_reference: The file that should be quarantined.
        reason: Why the file was quarantined.
        duplicate_group: Id shared by every copy of an exact duplicate, if that is the reason.
    """
    file_reference: ImageDatasetFileReference
    reason: tuple[str, ...]
    duplicate_group: Optional[str] = None


@dataclass(frozen=True)
class QuarantineReport:
    """
    What one stage found. Returned by every check loop and never acted on by the stage itself.

    Attributes:
        stage: Which stage produced the report.
        # checked: How many files the stage looked at.
        entries: The files it wants quarantined.
        # warnings: Notes that are logged but do not move anything.
    """
    stage: QuarantineStage
    # checked: int
    entries: tuple[QuarantineEntry, ...] = ()
    # warnings: tuple[str, ...] = ()


class DatasetDirFilesManager:
    # Finder recreates these constantly, so they are ignored rather than quarantined.
    IGNORED_FILENAMES: frozenset[str] = frozenset({".DS_Store"})  # TODO: what about other file systems like windows and linux? add thumbs.db LOLOL

    def __init__(
            self,
            images_raw_dir: Path,
    ) -> None:
        """
        DatasetDirFilesManager has 2 jobs:
        1. Validate the directory structure of the dataset
            - Ensure expected dirs exist in their expected structure
            - immediately stop the dataset ingestion pipeline if there are unexpected directories
              and let the user handle it (may involve manually moving data to correct location)

        2. Validate that only approved system files and accepted image filetypes exist in the expected directory
            structure from (1).
            - Anything that is (a) not an approved system file or (b) not an approved image file, is
              documented in a QuarentineEntry and these are collected into a QuarantineReport.

        Usage:
        Step 1:
            Init the object

        Step 2:
            Run validate_dir_structure() -> ValidateDirStructureResult

        Gate:
            if validate_dir_structure().has_problems: raise RuntimeError(f"unable to continue due to directory issues... {e}")

        Step 3:
            Run validate_filetypes() -> QuarantineReport

        The step 3 result is passed in to the QuarantineManager for handling (e.g. generating the csv manifest report, dry_run=False)
            and the actual logic for csv writing, file moving, extension lowering, etc., logic all exists there.

        Args:
            images_raw_dir: Path to the raw images directory.
        """
        self.images_raw_dir: Path = images_raw_dir

    def _create_image_raw_subdirectories(self) -> None:
        """
        Creates expected training dataset directory structure. exists ok is True
        """
        for label in ClassLabels:
            for source in SourceLabels:
                final_dir_path: Path = self.images_raw_dir / label.value / source.value
                final_dir_path.mkdir(parents=True, exist_ok=True)

    def _assert_images_raw_layout_is_valid(self) -> ValidateDirStructureResult:
        """
        if anything in the tree is not part of the expected layout, log it and raise out

        The expected layout is images_raw_dir/<label>/<source>/<files>. Anything else is flagged, because it
        would never be reached by the scan: at the root, any entry that is not a label directory; inside each
        label, any entry that is not a source directory; inside each source, any subdirectory.
        System files such as .DS_Store are ignored at every level.

        Raises:
            RuntimeError: If any unexpected entry is found. Every one is logged first.
        """
        unexpected_dir_events: list[UnexpectedDirEvent] = []

        # Root level: only label directories belong here.
        for item_path in sorted(self.images_raw_dir.iterdir()):
            if item_path.name in self.IGNORED_FILENAMES:
                continue
            if not (item_path.is_dir() and item_path.name in ClassLabels.as_set()):
                unexpected_dir_events.append(UnexpectedDirEvent(
                    directory=item_path,
                    reason=f"not an expected label directory.")
                )

        for label in ClassLabels:
            label_dir: Path = self.images_raw_dir / label.value

            # Label level: only source directories belong here.
            for item_path in sorted(label_dir.iterdir()):
                if item_path.name in self.IGNORED_FILENAMES:
                    continue
                if not (item_path.is_dir() and item_path.name in SourceLabels.as_set()):
                    unexpected_dir_events.append(UnexpectedDirEvent(
                        directory=item_path,
                        reason=f"not an expected source directory.")
                    )

            # Source level: files only, no subdirectories.
            for source in SourceLabels:
                for item_path in sorted((label_dir / source.value).iterdir()):
                    if item_path.is_dir():
                        unexpected_dir_events.append(UnexpectedDirEvent(
                            directory=item_path,
                            reason=f"a source directory should not have subdirectories.")
                        )

        # If anything is unexpected, log it and raise to stop until the user has corrected their dataset.
        return ValidateDirStructureResult(results=unexpected_dir_events)


    def validate_dir_structure(self) -> ValidateDirStructureResult:
        # 1. Make the expected dir structure
        self._create_image_raw_subdirectories()

        # 2. Check only the expected dir structure exists. log issues, return structured res object
        return self._assert_images_raw_layout_is_valid()


    def _iter_raw_files(self) -> Iterator[tuple[ClassLabels, SourceLabels, Path]]:
        """
        DOES NOT REMOVE NON-APPROVED IMAGE TYPES

        Yield every regular file under images_raw_dir/<label>/<source>/, skipping ignored system files.

        Missing directories are warned about and skipped. Anything that is not a regular file is warned about and
        skipped. Each directory is listed up front, so it is safe to move or rename files while iterating.

        Yields:
            A tuple of (label, source, file path), ordered by label, then source, then filename.
        """
        for label in ClassLabels:
            for source in SourceLabels:
                directory: Path = self.images_raw_dir / label.value / source.value
                if not directory.is_dir():
                    logger.warning(f"Expected directory is missing, skipping: {directory}")
                    continue

                for path in sorted(directory.iterdir()):
                    if path.name in self.IGNORED_FILENAMES:
                        continue
                    if not path.is_file():
                        logger.warning(f"Skipping entry that is not a regular file: {path}")
                        continue
                    yield label, source, path

##################### OK UP TO HERE #####################

    def _find_non_approved_files(self) -> QuarantineReport:
        """
        Run this after the directory scanner tool.

        Assumumption is that when running this method, the dir layout is ok.
        We're now going to inspect the file contents and tell non approved typed image files they gotsta go!
        """
        quarantine_entry_list: list[QuarantineEntry] = []
        # Skipping system files, show me all files in the approved directory tree.
        for file in self._iter_raw_files():
            label, source, path = file
            # TODO: what if suffix doesn't exist? that would raise attribute error, no?
            if path.suffix.lower() not in AcceptedImageFormats.as_tuple(include_dot=True):
                quarantine = QuarantineEntry(
                    reference=ImageDatasetFileReference(
                        images_raw_dir=self.images_raw_dir,
                        source=source,
                        label=label,
                        filename=path.name,
                    ),
                    reasons=(f"{label.value}/{source.value}/{path.name} is not of an acceptable dataset image type", ),
                    duplicate_group=None
                )
                quarantine_list.append(quarantine)

        return QuarantineReport(
            stage=QuarantineStage.FILESYSTEM__DIR_STRUCTURE_CHECK,
            entries=tuple(quarantine_list)
        )


    # So now at this point in the pipeline, we've estabilished:
    #   - the desired dir structure exists
    #   - no superfluous dirs or sub directories exist
    #   - we've identified non approved image files
    #   - if we quarantined all the non approved image files, and ignore system files,
    #     we're left with the actual dataset (as naive image files, yet to be evaluated)
    #   -


    def scan_dataset_get_quarantine_report(self) -> QuarantineReport:
        # 1. Ensure desired directory structure exists
        self._create_image_raw_subdirectories()

        # 2. Ensure the desired directory structure ONLY exists.
        #    Missing or superfluous directories will cause RAISE OUT here
        self._assert_images_raw_layout_is_valid()

        # 3. Now that we have the desired dataset directory structure,
        #    Let's find any non-system files that aren't image types we want
        #    and create a quarantine report for them
        non_approved_image_files: QuarantineReport = self._find_non_approved_files()

        logger.info(
            f"Found {len(non_approved_image_files.entries)} files to quarantine from the dataset directory"
        )
        return non_approved_image_files


    def check_file_extension_lowercases(self) -> None:
        for file in self._iter_raw_files():
            label, source, path = file
            if not path.suffix:
                logger.warning(f"File {path} does not have an extension, skipping")
                continue

            if path.suffix.lower() not in AcceptedImageFormats.as_tuple(include_dot=True):
                logger.warning(f"File {path} is not of image file of acceptable format, skipping")
                continue

            if path.suffix != path.suffix.lower():
                logger.warning(f"File {path} is an image file but the extension is not fully lowercase.")
                rename_file_with_lowercase_extension(
                    file_path=path,
                    dry_run=False
                )


    def gen_list_image_refs(self) -> list[ImageDatasetFileReference]:
        """
        RUN THIS METHOD AFTER clean_dataset()!!!

        THIS METHOD ALLOWS YOU TO GET A LIST OF ImageReference OBJECTS THAT MAP YOUR DATASET!
        Create this list of ImageReferences and pass that into the DatasetImageQualityManager class.

        with a clean raw images dataset directory, we can now generate our reference objects

        this output will then feed into the existing DataQualityManager class
        which performs checks on the actual image file and contents.
        We don't need to worry about that implementation here.

        Returns:
            One ImageReference per file under images_raw_dir/<label>/<source>/.
        """
        references: list[ImageDatasetFileReference] = [
            ImageDatasetFileReference(
                images_raw_dir=self.images_raw_dir,
                source=source,
                label=label,
                filename=path.name
            )
            for label, source, path in self._iter_raw_files()
        ]
        logger.info(f"Generated {len(references)} image references from {self.images_raw_dir}")
        return references
