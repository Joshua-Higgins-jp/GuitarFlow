from enum import StrEnum


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
    def as_tuple(cls) -> tuple[str, ...]:
        return tuple(member.value for member in cls)
