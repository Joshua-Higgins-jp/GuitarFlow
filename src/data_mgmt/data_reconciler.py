from dataclasses import dataclass
from datetime import datetime
from io import BytesIO
from pathlib import Path

from PIL import Image
from loguru import logger

from config.globals import (
    AcceptedImageFormats,
    ClassLabels,
    SourceLabels,
    StatusLabels,
)
from config.paths import RAW_DIR
from data_mgmt.image_database import DatabaseManager
from data_mgmt.image_properties import ImageProperties
from data_mgmt.image_record_model import ImageRecordModel
from utils.dt_timestamps import get_dt_now_jst
from utils.hashing import sha256_from_bytes

# ---------------------------------------------------------------------------
# Accepted file extensions for the filesystem scan, derived from the enum
# so there is one source of truth.
# ---------------------------------------------------------------------------
_ACCEPTED_EXTENSIONS: frozenset[str] = AcceptedImageFormats.as_frozen_set(include_dot=True)
logger.debug(f"Accepted image formats: {_ACCEPTED_EXTENSIONS}")


# ---------------------------------------------------------------------------
# Scan target
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ScanTarget:
    """
    A single directory to scan during reconciliation.

    Frozen because once built, a target should not be mutated.
    Constructed by build_scan_targets() from the cartesian product
    of the requested labels and sources.
    """
    directory: Path
    label: ClassLabels
    source: SourceLabels


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------
@dataclass
class ReconciliationResult:
    """
    Mutable summary of a single reconciliation run.

    Populated incrementally during the two-pass process, then logged
    at the end. All hash-level detail goes through loguru inline;
    this object captures the aggregated counts.
    """
    files_scanned: int = 0
    already_active: int = 0
    reactivated: int = 0
    newly_ingested: int = 0
    failed_ingestions: int = 0
    marked_missing: int = 0
    directories_scanned: int = 0
    directories_skipped: int = 0

    def log_summary(self) -> None:
        """Log the final reconciliation summary via loguru."""
        logger.info("=" * 60)
        logger.info("RECONCILIATION SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Directories scanned:    {self.directories_scanned}")
        logger.info(f"Directories skipped:    {self.directories_skipped}")
        logger.info(f"Files scanned:          {self.files_scanned}")
        logger.info(f"Already active:         {self.already_active}")
        logger.info(f"Reactivated:            {self.reactivated}")
        logger.info(f"Newly ingested:         {self.newly_ingested}")
        logger.info(f"Failed ingestions:      {self.failed_ingestions}")
        logger.info(f"Marked missing:         {self.marked_missing}")
        logger.info("=" * 60)


# ---------------------------------------------------------------------------
# Scan target builder
# ---------------------------------------------------------------------------
def build_scan_targets(
        labels: list[ClassLabels],
        sources: list[SourceLabels],
        raw_dir: Path = RAW_DIR,
) -> list[ScanTarget]:
    """
    Build a list of scan targets from the cartesian product of labels x sources.

    Only directories that exist on disk are included. Missing directories
    are logged as warnings and skipped — this is expected when a source
    has data for one class but not another.

    Args:
        labels: Which class labels to include (e.g. [ELECTRIC, ACOUSTIC]).
        sources: Which sources to include (e.g. [UNSPLASH, PIXABAY]).
        raw_dir: Root of the raw image data directory.

    Returns:
        List of ScanTarget objects for directories that exist on disk.
    """
    targets: list[ScanTarget] = []

    for label in labels:
        for source in sources:
            directory: Path = raw_dir / label.value / source.value

            if not directory.exists():
                logger.warning(f"Scan target does not exist, skipping: {directory}")
                continue

            if not directory.is_dir():
                logger.warning(f"Scan target is not a directory, skipping: {directory}")
                continue

            targets.append(ScanTarget(directory=directory, label=label, source=source))
            logger.debug(f"Scan target added: {directory}")

    logger.info(
        f"Built {len(targets)} scan targets from {len(labels)} labels x {len(sources)} sources"
    )
    return targets


# ---------------------------------------------------------------------------
# Reconciler
# ---------------------------------------------------------------------------
class Reconciler:
    """
    Two-pass filesystem <-> database reconciler.

    This is the single entry point for all image ingestion. There is no
    separate ingester — reconciliation IS ingestion. New images are added,
    known images are confirmed, and images that have disappeared from disk
    are marked missing.

    Pass 1 (filesystem -> DB):
        Walk every scan target directory. For each image file, compute its
        SHA-256 hash. If the hash is already known to the DB, confirm it
        active. If not, extract image properties and insert a new record.

    Pass 2 (DB -> filesystem, via set difference):
        Any hash present in the DB but NOT seen during Pass 1 is marked
        missing. No second filesystem walk is required.

    MVP note on nullable fields:
        source_id, image_url, search_query, and image_license are all stored
        as None for now. Source-specific handler functions (e.g. Unsplash URL
        generation, EXIF extraction for self-shot images) will be added later
        and plugged into _build_record() without touching the rest of this class.

    Usage:
        db_manager = ImageMetadataDBManager(db_path=Path("data/guitarflow.db"))
        reconciler = Reconciler(db_manager=db_manager)

        targets = build_scan_targets(
            labels=[ClassLabels.ELECTRIC, ClassLabels.ACOUSTIC],
            sources=[SourceLabels.UNSPLASH, SourceLabels.REDDIT]
        )

        result = reconciler.run(targets=targets)
        result.log_summary()
    """
    def __init__(self, db_manager: DatabaseManager, dry_run: bool = False) -> None:
        """
        Initialise the reconciler.

        Args:
            db_manager: Database manager instance for querying and updating records.
            dry_run: If True, the filesystem is scanned and logged but no database
                writes are performed. Useful for verifying what would be ingested
                before committing.
        """
        self.db_manager: DatabaseManager = db_manager
        self.dry_run: bool = dry_run

        if dry_run:
            logger.warning("Reconciler initialised in DRY RUN mode — no DB writes will occur.")

    def run(self, targets: list[ScanTarget], validate_paths: bool = False) -> ReconciliationResult:
        """
        Execute the two-pass reconciliation, with an optional third path-validation pass.

        Pass 1 (filesystem -> DB): hash-based upsert of all files on disk.
        Pass 2 (set difference):   mark any DB hash absent from disk as missing.
        Pass 3 (DB -> filesystem): row-by-row filename existence check for all
                                   active records. Catches the stale-filename bug
                                   where a hash is known but the file it points to
                                   has been renamed or deleted. Only runs when
                                   validate_paths=True.

        Args:
            targets: List of ScanTarget objects as returned by build_scan_targets().
            validate_paths: If True, run Pass 3 after Pass 2. Recommended whenever
                            you suspect manual file deletions or renames have occurred.
                            Safe to run on every reconciliation for sub-2000 datasets.

        Returns:
            ReconciliationResult with aggregated counts.
        """
        result = ReconciliationResult()
        now: datetime = get_dt_now_jst()

        # ---- Prepare: fetch all known hashes from DB ----
        db_hashes: set[str] = self.db_manager.get_all_hashes()
        logger.info(f"Loaded {len(db_hashes)} existing hashes from database")

        # ---- Pass 1: filesystem -> DB ----
        disk_hashes: set[str] = set()

        for target in targets:
            image_files: list[Path] = self._collect_image_files(directory=target.directory)
            result.directories_scanned += 1

            logger.info(f"Scanning {target.directory} — {len(image_files)} image files found")

            for filepath in image_files:
                result.files_scanned += 1

                try:
                    raw_bytes: bytes = filepath.read_bytes()
                    image_hash: str = sha256_from_bytes(data=raw_bytes)

                except Exception as e:
                    logger.error(f"Failed to read/hash {filepath.name}: {e}")
                    result.failed_ingestions += 1
                    continue

                disk_hashes.add(image_hash)

                if image_hash in db_hashes:
                    if not self.dry_run:
                        self.db_manager.confirm_active(
                            image_hash=image_hash,
                            last_seen=now,
                            current_filename=filepath.name,  # ilename drift check
                        )

                    result.already_active += 1
                    logger.debug(f"Confirmed active: {image_hash[:12]}...")

                else:
                    try:
                        record: ImageRecordModel = self._build_record(
                            filepath=filepath,
                            target=target,
                            raw_bytes=raw_bytes,
                            image_hash=image_hash,
                            now=now,
                        )
                        if not self.dry_run:
                            self.db_manager.insert(record)
                            db_hashes.add(image_hash)

                        result.newly_ingested += 1

                        logger.info(
                            f"{'[DRY RUN] Would ingest' if self.dry_run else 'Ingested'}: "
                            f"{filepath.name} ({image_hash[:12]}...)"
                        )

                    except Exception as e:
                        logger.error(f"Failed to ingest {filepath.name}: {e}")
                        result.failed_ingestions += 1

        # ---- Pass 2: set difference -> mark missing ----
        missing_hashes: set[str] = db_hashes - disk_hashes
        if missing_hashes:
            logger.info(f"Pass 2: {len(missing_hashes)} hashes in DB but not on disk")
            if not self.dry_run:
                result.marked_missing = self.db_manager.bulk_mark_missing(missing_hashes)
            else:
                result.marked_missing = len(missing_hashes)
                logger.info(f"[DRY RUN] Would mark {result.marked_missing} records as missing")
        else:
            logger.info("Pass 2: no missing images detected")

        # ---- Pass 3: row-by-row filename existence check ----
        if validate_paths:
            logger.info("Pass 3: validating file paths for all active records...")
            stale_hashes: set[str] = self._find_stale_path_hashes()

            if stale_hashes:
                logger.warning(f"Pass 3: {len(stale_hashes)} active records point to non-existent files")
                if not self.dry_run:
                    result.marked_missing += self.db_manager.bulk_mark_missing(stale_hashes)
                else:
                    result.marked_missing += len(stale_hashes)
                    logger.info(f"[DRY RUN] Would mark {len(stale_hashes)} stale-path records as missing")
            else:
                logger.info("Pass 3: all active records have valid file paths")

        result.log_summary()
        return result

    def _find_stale_path_hashes(self) -> set[str]:
        """
        Check every active DB row's expected file path and return hashes whose
        file no longer exists on disk.

        Reconstructs the expected path as RAW_DIR / label / source / filename,
        mirroring the structure that build_scan_targets() operates on.

        Returns:
            Set of image_hash values whose expected file path does not exist on disk.
        """
        stale: set[str] = set()
        records: list[dict] = self.db_manager.get_all_active_filename_records()

        for record in records:
            expected_path: Path = self._build_expected_path(
                label=record["label"],
                source=record["source"],
                filename=record["filename"],
            )
            if not expected_path.exists():
                logger.warning(
                    f"Stale path — hash {record['image_hash'][:12]}... "
                    f"points to missing file: {expected_path}"
                )
                stale.add(record["image_hash"])

        return stale

    @staticmethod
    def _build_expected_path(label: str, source: str, filename: str) -> Path:
        """
        Reconstruct the expected on-disk path for a given DB record.

        Args:
            label: The class label value (e.g. electric).
            source: The source label value (e.g. unsplash).
            filename: The bare filename stored in the DB.

        Returns:
            Absolute Path to where the file should exist under RAW_DIR.
        """
        return RAW_DIR / label / source / filename

    @staticmethod
    def _build_record(
            filepath: Path,
            target: ScanTarget,
            raw_bytes: bytes,
            image_hash: str,
            now: datetime,
    ) -> ImageRecordModel:
        """
        Build an ImageRecordModel for a newly discovered image file.

        Opens the PIL Image once to extract properties via ImageProperties.from_pil().
        The PIL Image is closed immediately after — raw_bytes is the durable
        reference, not the Image object.

        Nullable fields (source_id, image_url, search_query, image_license) are
        set to None for MVP. Source-specific handlers will be plugged in here later.

        Args:
            filepath: Path to the image file on disk.
            target: The ScanTarget this file was found under (carries label and source).
            raw_bytes: Raw bytes of the image file, already read by the caller.
            image_hash: SHA-256 hex digest of raw_bytes, already computed by the caller.
            now: Timestamp to use for ingested_at and last_seen (JST).

        Returns:
            A fully populated ImageRecordModel ready to insert into the database.

        Raises:
            PIL.UnidentifiedImageError: If the file cannot be opened as an image.
            ValueError: If the image format is not in AcceptedImageFormats.
        """
        # in the image_properties.py method, I believe it alludes to this Image.Image object already existing externally/
        # would that make this redundant?
        with Image.open(BytesIO(raw_bytes)) as img:
            props: ImageProperties = ImageProperties.from_pil(
                img=img,
                filesize_bytes=len(raw_bytes),
                image_hash=image_hash,
            )

        acquired_at: datetime = datetime.fromtimestamp(filepath.stat().st_mtime)

        return ImageRecordModel(
            image_hash=image_hash,
            source=target.source,
            label=target.label,
            status=StatusLabels.ACTIVE,
            source_id=None,       # TODO: plug in source-specific ID extractor
            image_url=None,       # TODO: plug in source-specific URL generator
            filename=filepath.name,
            search_query=None,    # TODO: pass through from caller if needed
            acquired_at=acquired_at,
            ingested_at=now,
            last_seen=now,
            image_license=None,   # TODO: plug in per-source license string
            width=props.width,
            height=props.height,
            filesize_bytes=props.filesize_bytes,
            image_format=props.image_format,
            num_channels=props.num_channels,
        )

    @staticmethod
    def _collect_image_files(directory: Path) -> list[Path]:
        """
        Collect all accepted image files in a directory (flat scan, non-recursive).

        Only files whose extension matches AcceptedImageFormats are included.
        AcceptedImageFormats is the single source of truth for accepted extensions.

        Args:
            directory: Directory to scan.

        Returns:
            Sorted list of image file paths.
        """
        return sorted(
            f for f in directory.iterdir()
            if f.is_file() and f.suffix.lower() in _ACCEPTED_EXTENSIONS
        )