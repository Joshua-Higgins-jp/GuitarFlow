from datetime import datetime
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field

from config.globals import ClassLabels, SourceLabels, StatusLabels, AcceptedImageFormats


class ImageRecordModel(BaseModel):
    """
    Single image metadata record - this is for LOCAL images.

    Identity is the SHA-256 hash of the raw image bytes - not a UUID.
    This means the same image content is always the same record, regardless
    of filename, location, or how many times it disappears and reappears.

    filepath is never stored. It is always derived via resolve_path() from
    label, source, image_hash, and file_extension. This makes paths resilient
    to environment changes - only DATA_DIR needs to be configured externally.
    """
    image_hash: str                    # SHA-256 hex digest - the identity of this image
    source: SourceLabels
    label: ClassLabels
    status: StatusLabels

    source_id: Optional[str] = None
    image_url: Optional[str] = None
    filename: str
    search_query: Optional[str] = None  # the search query on source that procured the image

    # Collection metadata
    acquired_at: datetime  # When the image entered our system (file mtime, download time, photo taken time)
    ingested_at: datetime  # When this metadata record was created (now())

    image_license: Optional[str] = None

    # Image properties (just have to be positive, non-zero)
    width: int = Field(..., gt=0)
    height: int = Field(..., gt=0)
    filesize_bytes: int = Field(..., gt=0)
    image_format: AcceptedImageFormats
    num_channels: int = Field(..., ge=1, le=4)  # 1 channel is BW. probably, we should filter these out? 3 is RGB, 4 adds alpha channel

    def resolve_path(
            self,
            raw_dir: Path
    ) -> Path:
        """
        Derives the expected filesystem path from semantic fields alone.

        No hardcoded paths in database. Rather, we derive them from image metadata fields.
        There is no actual validation the image exists at this path. That should be checked by the caller.

        Args:
            raw_dir: Root of the raw image data directory (e.g. DATA_DIR / raw).

        Returns:
            Expected path e.g. raw/electric/unsplash/abc123.jpg
        """
        return raw_dir / self.label.value / self.source.value / f"{self.filename}"
