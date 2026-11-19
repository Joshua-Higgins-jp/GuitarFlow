from enum import StrEnum


class StatusLabels(StrEnum):
    """
    reconciler status: active or missing.
    missing can imply delete, but we don't speculate; just confirm "not present aka missing"
    """
    ACTIVE = "active"
    MISSING = "missing"

    @classmethod
    def as_tuple(cls) -> tuple[str, ...]:
        return tuple(member.value for member in cls)
