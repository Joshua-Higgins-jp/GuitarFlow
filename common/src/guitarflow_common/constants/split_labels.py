from enum import StrEnum


class SplitLabels(StrEnum):
    """
    test, train, or split
    """
    TEST = "test"
    TRAIN = "train"
    VAL = "val"

    @classmethod
    def as_tuple(cls) -> tuple[str, ...]:
        return tuple(member.value for member in cls)
