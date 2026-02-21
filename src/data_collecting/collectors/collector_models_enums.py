from datetime import datetime
from enum import Enum

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

    image_id: str  # id from the image source, not internal id
    image_url: str
    search_query: str
    searched_at: datetime
    license: str

    width: int = None
    height: int = None
    filesize_KB: int= None
    downloaded_at: datetime = None
