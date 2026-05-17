from dataclasses import dataclass, field
from hashlib import sha256
from io import BytesIO
from pathlib import Path
from typing import Optional

from PIL import Image

from config.globals import AcceptedImageFormats
from utils.dt_timestamps import get_dt_now_utc


@dataclass
class ImageMetadata:
    """
    Physical and structural metadata extracted from an input image.

    Intended to be constructed via the from_uploaded or from_pil classmethods
    rather than directly. Used to populate the input fields of InferenceEvent
    before shipping to Datadog.
    """
    original_filename: str
    filename_pii_safe: str
    image_hash: str
    width_px: int
    height_px: int
    image_format: str
    file_size_bytes: int
    num_channels: int

    _image: Optional[Image.Image] = field(default=None, init=False, repr=False)

    @property
    def image(self) -> Image.Image:
        """
        Return the decoded PIL Image.

        Populated during construction via the classmethods. The property
        exists to expose the private field publicly as read-only.
        """
        if self._image is None:
            raise RuntimeError("ImageMetadata has no image loaded. Use from_bytes or from_local_file to construct.")
        return self._image

    @staticmethod
    def _gen_image_hash(image_raw_bytes: bytes) -> str:
        """
        Returns the image bytes as a unique hash string.

        Args:
            image_raw_bytes: The image as raw bytes
        """
        return sha256(image_raw_bytes).hexdigest()

    @staticmethod
    def _make_pii_safe_filename() -> str:
        """
        Generate a timestamp-based filename that contains no PII or metadata from the source image.
        """
        return get_dt_now_utc().strftime("%Y_%m_%d__%H_%M_%S")

    @classmethod
    def from_bytes(
            cls,
            raw_bytes: bytes,
            original_filename: Optional[str] = None,
    ) -> "ImageMetadata":
        """
        Build ImageMetadata from raw image bytes.

        Accepts any bytes-like source — a file read, a Streamlit UploadedFile
        that has already been drained, a network payload, etc.
        The caller is responsible for reading and buffering the bytes; this
        method does not touch any IO objects.

        Args:
            raw_bytes: The full contents of the image file as a plain bytes object.
            original_filename: The original filename or path stem if known. Used as a
                fallback to infer image format when PIL cannot determine it from the
                byte stream (e.g. some WEBP files). Stored as-is on the dataclass field.
        """
        image: Image.Image = Image.open(fp=BytesIO(raw_bytes))

        format_fallback: str = (
            Path(original_filename).suffix.lstrip(".").upper()
            if original_filename
            else "unknown"
        )

        instance = cls(
            original_filename=original_filename or "",
            filename_pii_safe=cls._make_pii_safe_filename(),
            image_hash=cls._gen_image_hash(raw_bytes),
            width_px=image.width,
            height_px=image.height,
            image_format=image.format or format_fallback,
            file_size_bytes=len(raw_bytes),
            num_channels=len(image.getbands()),
        )
        instance._image = image
        return instance

    @classmethod
    def from_local_file(cls, image_file_path: Path) -> "ImageMetadata":
        """
        Build ImageMetadata from a local image file on disk.

        Reads the file at the given path into bytes and delegates to from_bytes,
        preserving the original filename separately from the PII-safe generated name.

        Args:
            image_file_path: Absolute or relative path to the image file on disk.
        """
        return cls.from_bytes(
            raw_bytes=image_file_path.read_bytes(),
            original_filename=image_file_path.name,
        )

    def is_valid(self) -> bool:
        """
        Return True if the image passes basic sanity checks.

        Checks that dimensions are positive, channel count is within the
        accepted range, and the format is one the model was trained on.
        """
        return (
            self.width_px > 0
            and self.height_px > 0
            and 1 <= self.num_channels <= 4
            and self.image_format.upper() in AcceptedImageFormats.as_frozen_set()
        )
