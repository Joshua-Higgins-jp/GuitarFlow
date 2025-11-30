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

    source_id: str
    url: str
    search_query: str
    license: str
    downloaded_at: datetime

    width: Optional[int] = None
    height: Optional[int] = None

    filesize: Optional[int] = None
