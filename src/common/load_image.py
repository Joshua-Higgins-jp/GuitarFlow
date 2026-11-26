import hashlib
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Optional, Self

from PIL import Image, ImageOps


@dataclass(frozen=True, eq=False)
class LoadImage:
    """
    Everything the checks need, gathered from one open of the file.

    Metadata is reported as found in the file, so callers can filter on it later. The file itself is never changed.

    Attributes:
        pixels: Fully decoded RGB image with the EXIF orientation already applied.
        filepath: Path to the image file.
        format: Format PIL detected from the file content, for example "JPEG". None if unknown. Camera JPEGs with a
            multi-picture header are reported as "MPO".
        mode: PIL mode of the image as stored in the file, read before the conversion to RGB. Describes the source,
            not pixels, which is always "RGB". Common values:
                "1"      1-bit black and white
                "L"      8-bit greyscale
                "LA"     8-bit greyscale with alpha
                "P"      8-bit palette (indexed colour, for example GIF and many PNGs)
                "RGB"    8-bit per channel colour
                "RGBA"   8-bit per channel colour with alpha
                "CMYK"   print-style colour (common in JPEGs from design software)
                "I;16"   16-bit greyscale (for example 16-bit PNG and TIFF, medical and scientific images)
                "I"      32-bit integer greyscale
                "F"      32-bit float greyscale
                "RGBa"   RGB with premultiplied alpha ("La" is the greyscale equivalent)
            Modes "I;16", "I" and "F" hold more than 8 bits per pixel, so converting them straight to RGB can give badly
            wrong brightness. "CMYK" and "P" are converted by PIL's default rules, with no colour profile applied.
        n_frames: Number of frames in the file (1 for an ordinary still image).
        has_transparency: True if the file has an alpha channel or a transparency key (palette or colour key).
        exif_orientation: Orientation tag stored in the file (1 if missing or invalid, valid values are 1 to 8).
    """
    pixels: Image.Image
    filepath: Path
    format: Optional[str]
    mode: str
    n_frames: int
    has_transparency: bool
    exif_orientation: int

    @classmethod
    def from_path(cls, filepath: Path) -> Self:
        """
        Open an image once and gather everything the checks need.

        Metadata is reported as found in the file, so callers can filter on it later. The file is never modified.

        Args:
            filepath: Path to the image file.

        Returns:
            A LoadedImage holding RGB pixels (EXIF orientation applied) and the file metadata.

        Raises:
            FileNotFoundError: If the file does not exist.
            PIL.UnidentifiedImageError: If PIL cannot recognise the file as an image.
            OSError: If the file is truncated or corrupt.
            PIL.Image.DecompressionBombError: If the image is far larger than Image.MAX_IMAGE_PIXELS.
        """
        _EXIF_ORIENTATION_TAG: int = 0x0112

        with Image.open(filepath) as image:
            image.load()  # forces a complete decode, so truncated files fail here

            raw_orientation = image.getexif().get(_EXIF_ORIENTATION_TAG, 1)
            orientation: int = raw_orientation if isinstance(raw_orientation, int) and 1 <= raw_orientation <= 8 else 1

            has_alpha_band: bool = any(band in ("A", "a") for band in image.getbands())
            has_transparency: bool = has_alpha_band or "transparency" in image.info

            return cls(
                pixels=ImageOps.exif_transpose(image).convert("RGB"),
                filepath=filepath,
                format=image.format,
                mode=image.mode,
                n_frames=getattr(image, "n_frames", 1),
                has_transparency=has_transparency,
                exif_orientation=orientation,
            )

    @cached_property
    def pixel_sha256(self) -> str:
        """
        Hash of the image dimensions and RGB pixel values, ignoring how the file stores them.

        Two files with the same size and the same decoded pixels give the same digest, whatever their format,
        compression or metadata. The EXIF orientation is already applied, so a file tagged as rotated matches a copy
        that was rotated for real. This finds exact duplicates only. It is computed on first access and then cached,
        because hashing a large image is slow.

        Use this for exact duplicate finding. Similar images, cropped images, re-encoded images will still
        fail, so this is literally for like if you accidentily put images in 2 dirs, or into different labsls etc.

        Returns:
            A 64 character lowercase hex digest.
        """
        digest = hashlib.sha256()
        digest.update(f"{self.pixels.width}x{self.pixels.height}:RGB:".encode("ascii"))
        digest.update(self.pixels.tobytes())
        return digest.hexdigest()
