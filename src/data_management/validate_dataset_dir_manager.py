from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Iterator

from loguru import logger

from common.constants.class_labels import ClassLabels
from common.constants.source_labels import SourceLabels
from data_management.dataset_file_reference import DatasetFileReference


class DatasetValidationStage(StrEnum):
    FILESYSTEM = "filesystem"
    IMAGE = "image"
    DUPLICATE = "duplicate"  # TODO: actually use this or remove it. pls chk


@dataclass(frozen=True)
class DirStrucValError:
    """
    stores the results of a directory scan issue.
    Attributes:
        directory: Path to the unexpected directory.
        reason: the plain text reason why the directory was not expected.
    """
    directory: Path
    reason: str


@dataclass(frozen=True)
class ValidateDirStructureResult:
    """
    contains a list of directory structure validation errors found while validating the
    training dataset image dir structure
    """
    results: list[DirStrucValError]

    @property
    def has_problems(self) -> bool:
        return self.num_problems > 0

    @property
    def num_problems(self) -> int:
        return len(self.results)

    def log_results(self) -> None:
        for result in self.results:
            logger.warning(f"Directory `{result.directory}` is {result.reason}")

    def runtime_error_str(self) -> str:
        return (
            f"{self.num_problems} unexpected directory entries found. "
            f"Check logs and rectify, then try again."
        )


@dataclass(frozen=True)
class TargetsAndNonTargetsSets:
    """
    stores a set of DatasetFileReference objects for targets and non-targets.
    targets are the is_target_image_filetype True (i.e. images that are in AcceptableImageFormats)
    and non-targets are the is_target_image_filetype False (i.e. images that are not in AcceptableImageFormats)
    """
    targets_set: frozenset[DatasetFileReference]
    non_targets_set: frozenset[DatasetFileReference]

    @property
    def num_targets(self) -> int:
        return len(self.targets_set)

    @property
    def num_non_targets(self) -> int:
        return len(self.non_targets_set)

    @property
    def total_files(self) -> int:
        return self.num_targets + self.num_non_targets

    def log_results(self) -> None:
        logger.info(
            f"Total non-system files in dataset = {self.total_files}; "
            f"targets={self.num_targets}; "
            f"non-targets={self.num_non_targets}"
        )


class DatasetValidationDirFilesManager:
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
        unexpected_dir_events: list[DirStrucValError] = []

        # Root level: only label directories belong here.
        for item_path in sorted(self.images_raw_dir.iterdir()):
            if item_path.name in self.IGNORED_FILENAMES:
                continue
            if not (item_path.is_dir() and item_path.name in ClassLabels.as_set()):
                unexpected_dir_events.append(DirStrucValError(
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
                    unexpected_dir_events.append(DirStrucValError(
                        directory=item_path,
                        reason=f"not an expected source directory.")
                    )

            # Source level: files only, no subdirectories.
            for source in SourceLabels:
                for item_path in sorted((label_dir / source.value).iterdir()):
                    if item_path.is_dir():
                        unexpected_dir_events.append(DirStrucValError(
                            directory=item_path,
                            reason=f"a source directory should not have subdirectories.")
                        )

        # If anything is unexpected, log it and raise to stop until the user has corrected their dataset.
        return ValidateDirStructureResult(results=unexpected_dir_events)

    def _iter_raw_files(self) -> Iterator[tuple[ClassLabels, SourceLabels, Path]]:
        """
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

    def _gen_dataset_all_file_refs(self) -> frozenset[DatasetFileReference]:
        dataset_file_refs: set[DatasetFileReference] = set()
        for label, source, path in self._iter_raw_files():
            dataset_file_refs.add(
                DatasetFileReference(
                    images_raw_dir=self.images_raw_dir,
                    label=label,
                    source=source,
                    filename=path.name
                )
            )
        return frozenset(dataset_file_refs)

    def gen_target_sets(self) -> TargetsAndNonTargetsSets:
        # 0. confirm images raw directory structure is valid
        self._create_image_raw_subdirectories()

        val_res: ValidateDirStructureResult = self._assert_images_raw_layout_is_valid()
        val_res.log_results()
        if val_res.has_problems:
            raise RuntimeError(val_res.runtime_error_str())

        # 1. init empty return sets
        targets_set: set[DatasetFileReference] = set()
        non_targets_set: set[DatasetFileReference] = set()

        # 2. generate all file refs from the image dataset
        dataset_all_file_refs: frozenset[DatasetFileReference] = self._gen_dataset_all_file_refs()

        # 3. split them by target / non-target
        for file_ref in dataset_all_file_refs:
            targets_set.add(file_ref) if file_ref.is_target_image_filetype else non_targets_set.add(file_ref)

        # 4. return ground truth targets and non targets sets
        return TargetsAndNonTargetsSets(
            targets_set=frozenset(targets_set),
            non_targets_set=frozenset(non_targets_set)
        )
