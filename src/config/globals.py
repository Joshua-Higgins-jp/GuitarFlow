from enum import StrEnum
from typing import Tuple


class ClassLabels(StrEnum):
    """Image is labeled as either electric or acoustic."""
    ACOUSTIC = "acoustic"
    ELECTRIC = "electric"

    # NOT_GUITAR = "not_guitar"

    @classmethod
    def as_tuple(cls) -> Tuple[str, ...]:
        return tuple(member.value for member in cls)
