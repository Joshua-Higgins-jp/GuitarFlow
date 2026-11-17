from enum import StrEnum
from typing import Tuple


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
