from dataclasses import dataclass
from pathlib import Path
from typing import Self

from common.load_image import LoadImage


@dataclass(frozen=True)
class ImageMetadata:
    """
    Facts about one image file, without the decoded pixels.

    Knows nothing about datasets, labels or checks, so it can describe any image file. Build it while a LoadImage is
    loaded, then let the LoadImage go: the facts are kept, the pixels are freed.

    Attributes:
        filepath: Path to the image file.
        pixel_hash: SHA-256 of the decoded pixels and dimensions (LoadImage.pixel_sha256).
        width: Width in pixels after EXIF orientation is applied.
        height: Height in pixels after EXIF orientation is applied.
        mode: PIL mode as stored in the file, before conversion to RGB.
        image_format: Format PIL detected from the file content, with MPO reported as JPEG and "UNKNOWN" if PIL
            reported none.
        exif_orientation: EXIF orientation tag stored in the file, 1 to 8 (1 if missing).
        filesize_bytes: Size of the file on disk.
    """
    filepath: Path
    pixel_hash: str
    width: int
    height: int
    mode: str
    image_format: str
    exif_orientation: int
    filesize_bytes: int

    @classmethod
    def from_load_image(cls, load_image: LoadImage) -> Self:
        """Copy the file facts out of a loaded image.

        Args:
            load_image: The loaded image.

        Returns:
            The metadata, holding no reference to the decoded pixels.
        """
        image_format = "JPEG" if load_image.format == "MPO" else (load_image.format or "UNKNOWN")
        return cls(
            filepath=load_image.filepath,
            pixel_hash=load_image.pixel_sha256,
            width=load_image.pixels.width,
            height=load_image.pixels.height,
            mode=load_image.mode,
            image_format=image_format,
            exif_orientation=load_image.exif_orientation,
            filesize_bytes=load_image.filepath.stat().st_size,
        )

    @classmethod
    def from_path(cls, filepath: Path) -> Self:
        """Load an image file and copy out its facts.

        Args:
            filepath: Path to the image file.

        Returns:
            The metadata of the file.

        Raises:
            FileNotFoundError: If the file does not exist.
            PIL.UnidentifiedImageError: If PIL cannot recognise the file as an image.
            OSError: If the file is truncated or corrupt.
            PIL.Image.DecompressionBombError: If the image is far larger than Image.MAX_IMAGE_PIXELS.
        """
        return cls.from_load_image(load_image=LoadImage.from_path(filepath=filepath))