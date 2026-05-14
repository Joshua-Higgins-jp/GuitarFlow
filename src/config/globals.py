from enum import StrEnum
from typing import Tuple


_SHOW_LOGS = False


class ClassLabels(StrEnum):
    """Image is labeled as either electric or acoustic."""
    ACOUSTIC = "acoustic"
    ELECTRIC = "electric"
    NOT_GUITAR = "not_guitar"

    @classmethod
    def as_tuple(cls) -> Tuple[str, ...]:
        return tuple(member.value for member in cls)


class SplitLabels(StrEnum):
    TEST = "test"
    TRAIN = "train"
    VAL = "val"

    @classmethod
    def as_tuple(cls) -> Tuple[str, ...]:
        return tuple(member.value for member in cls)


class SourceLabels(StrEnum):
    MANUFACTURER_SITE = "manufacturer_site"
    MYSELF = "myself"
    PIXABAY = "pixabay"
    REDDIT = "reddit"
    UNSPLASH = "unsplash"

    @classmethod
    def as_tuple(cls) -> Tuple[str, ...]:
        return tuple(member.value for member in cls)
