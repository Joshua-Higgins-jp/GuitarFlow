from enum import StrEnum
from typing import Tuple


_SHOW_LOGS = False


class ClassLabels(StrEnum):
    """
    Image is labeled as either electric, acoustic, or "not_guitar"
    """
    ACOUSTIC = "acoustic"
    ELECTRIC = "electric"
    NOT_GUITAR = "not_guitar"

    @classmethod
    def as_tuple(cls) -> Tuple[str, ...]:
        return tuple(member.value for member in cls)


class SplitLabels(StrEnum):
    """
    test, train, or split
    """
    TEST = "test"
    TRAIN = "train"
    VAL = "val"

    @classmethod
    def as_tuple(cls) -> Tuple[str, ...]:
        return tuple(member.value for member in cls)


class SourceLabels(StrEnum):
    """
    currently 5 sources: manufacturer, myself, pixabay, reddit, unsplash, mixed
    """
    MANUFACTURER_SITE = "manufacturer_site"
    MYSELF = "myself"
    PIXABAY = "pixabay"
    REDDIT = "reddit"
    UNSPLASH = "unsplash"
    MIXED = "mixed"

    @classmethod
    def as_tuple(cls) -> Tuple[str, ...]:
        return tuple(member.value for member in cls)


class StatusLabels(StrEnum):
    """
    reconciler status: active or missing.
    missing can imply delete, but we don't speculate; just confirm "not present aka missing"
    """
    ACTIVE = "active"
    MISSING = "missing"

    @classmethod
    def as_tuple(cls) -> Tuple[str, ...]:
        return tuple(member.value for member in cls)


class AcceptedImageFormats(StrEnum):
    """
    must be uppercase please: JPEG, JPG, PNG, WEBP
    """
    JPEG = "jpeg"
    JPG = "jpg"
    PNG = "png"
    WEBP = "webp"

    @classmethod
    def as_tuple(cls) -> Tuple[str, ...]:
        return tuple(member.value for member in cls)

    @classmethod
    def as_frozen_set(cls, include_dot: bool = False) -> frozenset[str]:
        if include_dot:
            return frozenset(f".{member.value}" for member in cls)
        return frozenset(member.value for member in cls)
