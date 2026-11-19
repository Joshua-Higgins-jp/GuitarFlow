from enum import StrEnum


class ClassLabels(StrEnum):
    """
    Image is labeled as either electric, acoustic, or "not_guitar"
    """
    ACOUSTIC = "acoustic"
    ELECTRIC = "electric"
    NOT_GUITAR = "not_guitar"

    @classmethod
    def as_tuple(cls) -> tuple[str, ...]:
        return tuple(member.value for member in cls)

    @classmethod
    def as_list(cls) -> list[str]:
        return list(member.value for member in cls)
