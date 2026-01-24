from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel


class ImageSource(Enum):
    PIXABAY = "PIXABAY"
    UNSPLASH = "UNSPLASH"
    PEXELS = "PEXELS"


class ImageLabel(Enum):
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
