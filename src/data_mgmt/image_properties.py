from dataclasses import dataclass

from PIL import Image

from config.globals import AcceptedImageFormats


@dataclass(frozen=True)
class ImageProperties:
    """
    Physical and structural properties extracted from an image.

    Pure data — no IO, no file handles, no PIL Image reference stored.
    The caller owns the Image lifecycle; this class just reads the numbers
    off it and becomes an immutable snapshot.

    Constructed via from_pil(). The caller is responsible for:
        1. Reading the raw bytes (from disk, network, Streamlit upload, etc.)
        2. Computing the SHA-256 hash via sha256_from_bytes()
        3. Opening the PIL Image
        4. Passing all three into from_pil()

    This keeps ImageProperties free of IO concerns and means the PIL Image
    is only decoded once regardless of how many consumers need the metadata.
    """
    image_hash: str
    width: int
    height: int
    num_channels: int
    image_format: AcceptedImageFormats
    filesize_bytes: int

    @classmethod
    def from_pil(
            cls,
            img: Image.Image,
            filesize_bytes: int,
            image_hash: str
    ) -> "ImageProperties":
        """
        Extract physical properties from an already-opened PIL Image.

        Does not close or modify the Image. The caller retains full
        ownership of the Image lifecycle.

        Args:
            img: An open PIL Image. Must not be closed before calling this.
            filesize_bytes: Size of the original image bytes. Cannot be derived
                from PIL so the caller provides it.
            image_hash: Pre-computed SHA-256 hex digest of the raw image bytes.

        Returns:
            Frozen ImageProperties instance.

        Raises:
            ValueError: If the image format cannot be determined from PIL, or
            if the format is not in AcceptedImageFormats.
        """
        image_format = AcceptedImageFormats(img.format.lower()) if img.format else None
        return cls(
            image_hash=image_hash,
            width=img.width,
            height=img.height,
            num_channels=len(img.getbands()),
            image_format=image_format,
            filesize_bytes=filesize_bytes,
        )

    def is_valid(self) -> bool:
        """
        Return True if the image passes basic sanity checks.

        Checks that dimensions are positive, channel count is within the
        accepted range (1=greyscale, 3=RGB, 4=RGBA), and the format is
        one the model was trained on.

        Not raising on any of these is a design choice - if your processing handles invalid exceptions,
        that maybe a better use. Alternatively, you can call this is_valid() and raise if False.
        Same results, extra steps. Not designing landmines unless needed.
        """
        return (
            self.width > 0
            and self.height > 0
            and 1 <= self.num_channels <= 4
            and self.image_format in AcceptedImageFormats
        )
