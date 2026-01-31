from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel


class ImageSource(Enum):
    """image source is the site name for where the image originated"""
    PIXABAY = "PIXABAY"
    UNSPLASH = "UNSPLASH"
    PEXELS = "PEXELS"


class ImageLabel(Enum):
    """image is labeled as either electric or acoustic"""
    ELECTRIC = "ELECTRIC"
    ACOUSTIC = "ACOUSTIC"


class ImageMetadata(BaseModel):
    """Single image metadata"""
    source: ImageSource
    label: ImageLabel

    image_id: str
    image_url: str
    search_query: str
    searched_at: datetime
    license: Optional[str]

    width: Optional[int] = None
    height: Optional[int] = None
    filesize: Optional[int] = None
    is_downloaded: bool = False
    downloaded_at: Optional[datetime] = None
