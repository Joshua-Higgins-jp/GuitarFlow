from datetime import datetime
from pathlib import Path
from typing import Self

from pydantic import BaseModel

from common.constants.class_labels import ClassLabels
from common.constants.source_labels import SourceLabels
from common.constants.status_labels import StatusLabels
from common.image_metadata import ImageMetadata


class ImageRecordModel(BaseModel):
    """
    One row of the images table: a local image that has passed every dataset check at least once.

    The table lists the images the training pipeline may build datasets from. It is one wide table instead of several
    normalised ones on purpose. The project is small, and one row per image is easier to query and reason about.

    Identity:
        The primary key is pixel_hash, the SHA-256 of the decoded RGB pixels and dimensions after the EXIF orientation
        is applied (LoadImage.pixel_sha256). An earlier version hashed the raw file bytes. That gave the same picture
        a new identity whenever its metadata changed or it was re-saved losslessly, and it disagreed with the
        duplicate check about what counts as the same image. With the pixel hash, the record and the duplicate check
        agree. Filenames and folders can change, but the hash cannot.

    Paths:
        The file path is never stored. construct_path builds it from label, source and filename under images_raw, so
        only the images_raw root has to change between machines. A record does not guarantee the file exists; the
        training pipeline checks the path before it uses a record.

    When fields are written:
        Set once, on insert: pixel_hash, width, height, ingested_at. Width and height can never change for a given
        hash, because the hash includes the dimensions.
        Refreshed on every scan: label, source, filename, mode, image_format, exif_orientation, filesize_bytes,
        checks_version, status, last_seen_at. These describe where the file is now and how it is stored. A moved,
        renamed or re-saved file keeps its hash but can change any of them, and a refresh must never overwrite the
        set-once fields. A change of label or source moves the file to another folder and changes what the model
        learns from it, so the scan logs it before updating.

    Status:
        Only two values: ACTIVE if the image was found in the latest scan and passed every check, MISSING otherwise.
        A quarantined, deleted or moved-away file all look the same to a scan (not in images_raw), so finer states
        would only be guesses. The training pipeline uses ACTIVE rows only. Rows are never deleted, so an image that
        comes back continues its original record, with its original ingested_at.

    Deliberately not stored:
        - source_id, image_url, search_query, image_license, author: images are curated by hand, so the filename is
          the only provenance available, and columns that are always empty look like a broken feature.
        - acquired_at: file creation time is not portable (st_ctime on Linux is the last metadata change) and resets
          when a file is copied, so it is not a reliable fixed value. The filesystem still has it if needed.
        - A perceptual hash for near duplicates: the dataset is curated by hand, so near duplicates across splits are
          an accepted risk for now.
        - num_channels: replaced by mode, which says more ("CMYK" has four channels and no alpha).

    Validation:
        Images pass the dataset checks before a record is built, and the database enforces NOT NULL and positive
        sizes. So the model uses plain annotations with no Field constraints. Its main job is turning database text
        back into datetimes and enums when rows are read. mode and image_format are plain strings because they are
        metadata only, and an unexpected value should not stop a scan.

    Attributes:
        pixel_hash: 64 character lowercase hex SHA-256 of the decoded pixels. The identity of the image.
        width: Width in pixels after EXIF orientation is applied.
        height: Height in pixels after EXIF orientation is applied.
        ingested_at: When this record was first created, timezone aware (JST).
        label: Class label, taken from the image's folder.
        source: Where the image came from, taken from the image's folder.
        filename: Current filename inside the label and source folder, including the extension.
        mode: PIL mode as stored in the file, before conversion to RGB, for example "RGB" or "CMYK".
        image_format: Format PIL detected from the file content, with MPO reported as JPEG.
        exif_orientation: EXIF orientation tag stored in the file, 1 to 8 (1 if missing). Width, height and
            pixel_hash already include this rotation, so anything that loads the file for training must apply it too
            (LoadImage does).
        filesize_bytes: Size of the file on disk.
        checks_version: ImageCheckConfig.version of the scan that last checked this image.
        status: ACTIVE or MISSING, see Status above.
        last_seen_at: When a scan last found this image and it passed, timezone aware (JST).
    """
    # set once
    pixel_hash: str
    width: int
    height: int
    ingested_at: datetime

    # refreshed every scan
    label: ClassLabels
    source: SourceLabels
    filename: str
    mode: str
    image_format: str  # formerly AcceptedImageFormats, but this strict check could fail database IO
    exif_orientation: int
    filesize_bytes: int
    checks_version: str
    status: StatusLabels
    last_seen_at: datetime

    def construct_path(self, images_raw_dir: Path) -> Path:
        """Build the expected path of the image from its label, source and filename.

        The path is not checked. The caller must confirm the file exists before using it.

        Args:
            images_raw_dir: Root of the raw image directory, for example DATA_DIR / "images_raw".

        Returns:
            The expected path, for example images_raw/electric/unsplash/guitar_123.jpg.
        """
        return images_raw_dir / self.label.value / self.source.value / self.filename

    @classmethod
    def from_metadata(cls, metadata: ImageMetadata, checks_version: str, scanned_at: datetime) -> Self:
        """Build a record from the metadata of an image that passed every check.

        Label and source are read from the folders the file sits in (images_raw/<label>/<source>/<filename>), the
        same layout construct_path rebuilds. ingested_at and last_seen_at are both set to scanned_at. For an image
        that is already in the table, the upsert must keep the stored ingested_at and refresh only the other fields.

        Args:
            metadata: File facts of the image.
            checks_version: ImageCheckConfig.version of this scan.
            scanned_at: Timezone aware time of the scan, shared by every record the scan builds.

        Returns:
            The record, with status ACTIVE.

        Raises:
            ValueError: If scanned_at has no timezone, or a folder name is not a known label or source.
        """
        if scanned_at.tzinfo is None:
            raise ValueError("scanned_at must be timezone aware")

        filepath = metadata.filepath
        return cls(
            pixel_hash=metadata.pixel_hash,
            width=metadata.width,
            height=metadata.height,
            ingested_at=scanned_at,
            label=ClassLabels(filepath.parent.parent.name),
            source=SourceLabels(filepath.parent.name),
            filename=filepath.name,
            mode=metadata.mode,
            image_format=metadata.image_format,
            exif_orientation=metadata.exif_orientation,
            filesize_bytes=metadata.filesize_bytes,
            checks_version=checks_version,
            status=StatusLabels.ACTIVE,
            last_seen_at=scanned_at,
        )
