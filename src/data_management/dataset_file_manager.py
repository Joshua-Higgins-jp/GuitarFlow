import csv
from pathlib import Path
from typing import Iterator, Optional

from loguru import logger

from common.constants.accepted_image_formats import AcceptedImageFormats
from common.constants.class_labels import ClassLabels
from common.constants.session_id import SessionID
from data_management.image_reference import ImageReference
from common.constants.source_labels import SourceLabels
from common.utils.safe_move_file import safe_move_file


class DatasetFileManager:
    # Finder recreates these constantly, so they are ignored rather than quarantined.
    IGNORED_FILENAMES: frozenset[str] = frozenset({".DS_Store"})
    MANIFEST_COLUMNS: tuple[str, ...] = (
        "filename",
        "original_dir",
        "relative_path",
        "quarantined_filename",
        "reason"
    )
    def __init__(
            self,
            data_dir: Path,
            images_raw_dir: Path,
            quarantined_files_dir: Path,
            session_id: SessionID
    ) -> None:
        """
        These root directories are guaranteed IF they are assigned after a DirsPaths initialization.

        The SUBDIRECTORIES under images_raw_dir will be created/validated in this manager.

        i.e. data/datasets, data/images_raw, data/quarantine, data/mlflow-data exist after DirsPaths intialisaes
        BUT this class is focused on the substructure of images_raw_dir and quarantined_files_dir
        (at this point in time it will be flat anyway)

        Args:
            data_dir: Path to the data directory.
            images_raw_dir: Path to the raw images directory.
            quarantined_files_dir: Path to the quarantined files directory.
        """
        self.data_dir: Path = data_dir
        self.images_raw_dir: Path = images_raw_dir
        self.quarantined_files_dir: Path = quarantined_files_dir
        self.session_id: SessionID = session_id

        self._accepted_suffixes: tuple[str, ...] = AcceptedImageFormats.as_tuple(include_dot=True)

    def _create_image_raw_subdirectories(self) -> None:
        """creates expected directories. exists ok is True"""
        for label in ClassLabels:
            for source in SourceLabels:
                final_dir_path: Path = self.images_raw_dir / label.value / source.value
                final_dir_path.mkdir(parents=True, exist_ok=True)

    def _assert_images_raw_layout_is_valid(self) -> None:
        """if anything in the tree is not part of the expected layout, log it and raise out

        The expected layout is images_raw_dir/<label>/<source>/<files>. Anything else is flagged, because it
        would never be reached by the scan: at the root, any entry that is not a label directory; inside each
        label, any entry that is not a source directory; inside each source, any subdirectory.
        System files such as .DS_Store are ignored at every level.

        Raises:
            RuntimeError: If any unexpected entry is found. Every one is logged first.
        """
        expected_labels: set[str] = {label.value for label in ClassLabels}
        expected_sources: set[str] = {source.value for source in SourceLabels}
        unexpected: list[tuple[Path, str]] = []

        # Root level: only label directories belong here.
        for item in sorted(self.images_raw_dir.iterdir()):
            if item.name in self.IGNORED_FILENAMES:
                continue
            if not (item.is_dir() and item.name in expected_labels):
                unexpected.append((item, "not a known label directory"))

        for label in ClassLabels:
            label_dir: Path = self.images_raw_dir / label.value

            # Label level: only source directories belong here.
            for item in sorted(label_dir.iterdir()):
                if item.name in self.IGNORED_FILENAMES:
                    continue
                if not (item.is_dir() and item.name in expected_sources):
                    unexpected.append((item, "not a known source directory"))

            # Source level: files only, no subdirectories.
            for source in SourceLabels:
                for item in sorted((label_dir / source.value).iterdir()):
                    if item.is_dir():
                        unexpected.append((item, "subdirectory inside a source directory"))

        # If anything is unexpected, log it and raise to stop until the user has corrected their dataset.
        if unexpected:
            for path, why in unexpected:
                logger.warning(f"Unexpected item: {path.relative_to(self.images_raw_dir)} ({why})")
            raise RuntimeError(
                f"Found {len(unexpected)} unexpected items in the images raw directory. "
                f"Clean it up, and try again"
            )

    def _iter_raw_files(self) -> Iterator[tuple[ClassLabels, SourceLabels, Path]]:
        """Yield every regular file under images_raw_dir/<label>/<source>/, skipping ignored system files.

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

    def _resolve_quarantine_path(self, filename: str, reserved: set[str]) -> Path:
        """Choose a destination in quarantine that does not clash with an existing or already planned file.

        The original filename is used if free. Otherwise it becomes session_id__filename, and if that is taken too,
        session_id__2__filename, session_id__3__filename and so on.

        Args:
            filename: Original filename of the file being quarantined.
            reserved: Case-folded destination names already chosen during this run. The chosen name is added to it.

        Returns:
            A destination path in the quarantine directory.
        """
        candidate: Path = self.quarantined_files_dir / filename
        attempt = 1
        while candidate.exists() or candidate.name.casefold() in reserved:
            tag = self.session_id if attempt == 1 else f"{self.session_id}__{attempt}"
            candidate = self.quarantined_files_dir / f"{tag}__{filename}"
            attempt += 1

        reserved.add(candidate.name.casefold())
        return candidate

    def _write_manifest(self, rows: list[dict[str, str]], dry_run: bool) -> Optional[Path]:
        """Write the quarantine manifest CSV for this session.

        Args:
            rows: One dict per quarantined file, keyed by MANIFEST_COLUMNS.
            dry_run: If True, the file is named as a dry run manifest.

        Returns:
            Path of the CSV, or None if there were no rows.
        """
        if not rows:
            return None

        self.quarantined_files_dir.mkdir(parents=True, exist_ok=True)
        manifest_path: Path = (
            self.quarantined_files_dir / f"quarantine_manifest_{self.session_id.session_id_str}{'_dry_run' if dry_run else ''}.csv"
        )
        with manifest_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=self.MANIFEST_COLUMNS)
            writer.writeheader()
            writer.writerows(rows)
        return manifest_path

    def _scan_and_clean_images_raw_dir(self, dry_run: bool = False) -> Optional[Path]:
        """Quarantine every file under images_raw_dir that is not an acceptable image type.

        System files such as .DS_Store are ignored. Extensions are compared case-insensitively. The manifest records,
        for each file, its name, where it was, its path relative to images_raw_dir and what it is called in quarantine.
        If a move fails part way, the manifest still lists the files moved before the failure.

        Args:
            dry_run: If True, move nothing and write a dry run manifest of what would be moved.

        Returns:
            Path of the manifest CSV, or None if nothing needed quarantining.
        """
        rows: list[dict[str, str]] = []
        reserved: set[str] = set()
        action = "Would move" if dry_run else "Moved"

        logger.info(f"Acceptable image types: {sorted(self._accepted_suffixes)}")
        try:
            for _label, _source, path in list(self._iter_raw_files()):
                if path.suffix.lower() in self._accepted_suffixes:
                    continue

                reason: str = (
                    f"extension '{path.suffix}' is not an acceptable image type"
                    if path.suffix
                    else "no extension, so it cannot be an acceptable image type"
                )
                destination: Path = self._resolve_quarantine_path(path.name, reserved)
                if not dry_run:
                    safe_move_file(path, destination)

                rows.append(
                    {
                        "filename": path.name,
                        "original_dir": str(path.parent),
                        "relative_path": str(path.relative_to(self.images_raw_dir)),
                        "quarantined_filename": destination.name,
                        "reason": reason,
                    }
                )
                logger.info(f"{action} to quarantine ({reason}): {path} -> {destination}")
        # ... finally block unchanged ...
        finally:
            manifest_path: Optional[Path] = self._write_manifest(rows, dry_run)

        if manifest_path is None:
            logger.info("No files needed quarantining")
        else:
            logger.warning(f"{'DRY RUN: ' if dry_run else ''}{action} {len(rows)} files. Manifest: {manifest_path}")
        return manifest_path

    def _rename_file_with_lowercase_extension(self, file_path: Path, dry_run: bool = False) -> None:
        """only necessary for surviving image files. anything in quarantine can be whatever it was originally

        Args:
            file_path: Image file whose extension should be lower-cased.
            dry_run: If True, only log what would be renamed.

        Raises:
            FileExistsError: If a different file with the lower-cased name already exists.
        """
        lowered_path: Path = file_path.with_suffix(suffix=file_path.suffix.lower())
        if lowered_path.name == file_path.name:
            return

        # On a case-insensitive filesystem (macOS default) the lowered name resolves to this same file, which is fine.
        if lowered_path.exists() and not lowered_path.samefile(file_path):
            raise FileExistsError(f"Cannot lower-case extension, a different file already exists: {lowered_path}")

        logger.info(f"{'Would rename' if dry_run else 'Renaming'}: {file_path.name} -> {lowered_path.name}")
        if not dry_run:
            file_path.rename(lowered_path)

    def clean_dataset(self, dry_run: bool = False) -> Optional[Path]:
        """Make images_raw_dir contain only valid image files with lower-case extensions.

        Steps: create the expected subdirectories, raise if any unexpected subdirectory exists, quarantine
        non-image files, then lower-case the extensions of the surviving files.

        Args:
            dry_run: If True, move and rename nothing, and write a dry run manifest instead.

        Returns:
            Path of the quarantine manifest CSV, or None if nothing needed quarantining.

        Raises:
            RuntimeError: If any label/source directory contains a subdirectory.
            FileExistsError: If lower-casing an extension would overwrite a different file.
        """
        self._create_image_raw_subdirectories()
        self._assert_images_raw_layout_is_valid()

        manifest_path: Optional[Path] = self._scan_and_clean_images_raw_dir(dry_run=dry_run)

        # In a dry run the non-images are still present, so only look at files that would survive.
        for _label, _source, path in list(self._iter_raw_files()):
            if path.suffix.lower() in self._accepted_suffixes:
                self._rename_file_with_lowercase_extension(path, dry_run=dry_run)

        return manifest_path

    def gen_list_image_refs(self) -> list[ImageReference]:
        """
        with a clean raw images dataset directory, we can now generate our reference objects

        this output will then feed into the existing DataQualityManager class
        which performs checks on the actual image file and contents.
        We don't need to worry about that implementation here.

        Returns:
            One ImageReference per file under images_raw_dir/<label>/<source>/.
        """
        references: list[ImageReference] = [
            ImageReference(images_raw_dir=self.images_raw_dir, source=source, label=label, filename=path.name)
            for label, source, path in self._iter_raw_files()
        ]
        logger.info(f"Generated {len(references)} image references from {self.images_raw_dir}")
        return references
