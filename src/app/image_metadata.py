from dataclasses import dataclass
from hashlib import sha256
from io import BytesIO
from pathlib import Path
from typing import ClassVar

from PIL import Image
from streamlit.runtime.uploaded_file_manager import UploadedFile

from utils.dt_timestamps import get_dt_now_utc


@dataclass
class ImageMetadata:
    """
    Physical and structural metadata extracted from an input image.

    Intended to be constructed via the from_uploaded or from_pil classmethods
    rather than directly. Used to populate the input fields of InferenceEvent
    before shipping to Datadog.
    """
    filename: str
    image_hash: str
    width_px: int
    height_px: int
    image_format: str
    file_size_bytes: int
    num_channels: int

    ACCEPTED_FORMATS: ClassVar[frozenset[str]] = frozenset({"JPEG", "JPG", "PNG", "WEBP"})


    @staticmethod
    def _to_raw_bytes(image: Image.Image) -> bytes:
        """
        Encode a PIL Image to PNG bytes for hashing and size estimation.

        Parameters
        ----------
        image:
            The PIL Image to encode.
        """
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        return buffer.getvalue()

    @staticmethod
    def _make_filename() -> str:
        """
        Generate a timestamp-based filename that contains no user PII.
        """
        return get_dt_now_utc().strftime("%Y_%m_%d__%H_%M_%S")

    @classmethod
    def from_uploaded(cls, uploaded: UploadedFile) -> "ImageMetadata":
        """
        Build ImageMetadata from a Streamlit UploadedFile.

        Reads the file bytes once and derives all fields from them,
        then rewinds the buffer so the caller can still open the image.

        Args:
            uploaded: The UploadedFile object from st.file_uploader.
        """
        raw_bytes: bytes = uploaded.read()
        uploaded.seek(0)

        image: Image.Image = Image.open(fp=uploaded)
        uploaded.seek(0)

        return cls(
            filename=cls._make_filename(),
            image_hash=sha256(raw_bytes).hexdigest(),
            width_px=image.width,
            height_px=image.height,
            image_format=image.format or Path(uploaded.name).suffix.lstrip(".").upper(),
            file_size_bytes=len(raw_bytes),
            num_channels=len(image.getbands()),
        )

    @classmethod
    def from_pil(cls, image: Image.Image) -> "ImageMetadata":
        """
        Build ImageMetadata from an already-opened PIL Image.

        Used for the camera input path where there is no UploadedFile.
        File size is estimated from a PNG encode since camera frames
        have no original file on disk.

        Parameters
        ----------
        image:
            The PIL Image object.
        """
        raw_bytes: bytes = cls._to_raw_bytes(image)

        return cls(
            filename=cls._make_filename(),
            image_hash=sha256(raw_bytes).hexdigest(),
            width_px=image.width,
            height_px=image.height,
            image_format=image.format or "PNG",
            file_size_bytes=len(raw_bytes),
            num_channels=len(image.getbands()),
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
            and self.image_format.upper() in self.ACCEPTED_FORMATS
        )
