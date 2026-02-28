from datetime import datetime
from enum import Enum
from typing import Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class ImageSource(Enum):
    """Image source is the site name for where the image originated"""
    UNSPLASH = "UNSPLASH"
    REDDIT = "REDDIT"
    MANUFACTURER_SITE = "MANUFACTURER_SITE"
    MYSELF = "MYSELF"


class ImageLabel(Enum):
    """Image is labeled as either electric or acoustic"""
    ELECTRIC = "ELECTRIC"
    ACOUSTIC = "ACOUSTIC"


class ImageMetadata(BaseModel):
    """
    Single image metadata. Source agnostic - online download or taken by myself

    Optionals are only for when None is a VALID business case for a specific image.
    """
    # Internal management
    internal_id: UUID = Field(default_factory=uuid4)

    # Source information
    source: ImageSource
    label: ImageLabel

    # Source-specific identifiers
    source_id: Optional[str] = None  # Unsplash ID, Reddit post ID, etc.
    image_url: Optional[str] = None  # generate where possible
    original_filename: str

    # Collection metadata
    search_query: Optional[str] = None
    acquired_at: datetime  # When the image entered our system (file mtime, download time, photo taken time)
    ingested_at: datetime  # When this metadata record was created (now())
    license: Optional[str] = None

    # File properties
    width: int
    height: int
    filesize_bytes: int
