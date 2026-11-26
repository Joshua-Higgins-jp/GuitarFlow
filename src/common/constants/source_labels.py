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
    WALLPAPER_FLARE = "wallpaper_flare"

    @classmethod
    def as_tuple(cls) -> tuple[str, ...]:
        return tuple(member.value for member in cls)

    @classmethod
    def as_set(cls) -> set[str]:
        return set(member.value for member in cls)
