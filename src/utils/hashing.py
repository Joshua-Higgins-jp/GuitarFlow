from hashlib import sha256
from pathlib import Path


def sha256_from_bytes(data: bytes) -> str:
    """
    Compute the SHA-256 hex digest of raw bytes.

    Used for hashing in-memory image data (e.g. uploaded files,
    network payloads) where the bytes are already buffered.

    Args:
        data: Raw bytes to hash.

    Returns:
        64-character lowercase hex digest string.
    """
    return sha256(data).hexdigest()


def sha256_from_file(filepath: Path) -> str:
    """
    Compute the SHA-256 hex digest of a file on disk.

    Reads the entire file into memory. For the image sizes in this
    project (typically < 20 MB), this is fine. If you ever need to
    hash multi-GB files, swap to a chunked read.

    Args:
        filepath: Path to the file to hash.

    Returns:
        64-character lowercase hex digest string.

    Raises:
        FileNotFoundError: If filepath does not exist.
        IsADirectoryError: If filepath is a directory.
    """
    return sha256(filepath.read_bytes()).hexdigest()
