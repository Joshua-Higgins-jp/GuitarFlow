from pathlib import Path
from shutil import copy2


def safe_move_file(
        source_path: Path,
        destination_path: Path
) -> Path:
    """
    Move a file by copying it, confirming the copy, and only then deleting the source.

    If the copy does not check out, the bad copy is removed and the source is left untouched.

    Args:
        source_path: File to move.
        destination_path: Where it should end up. Must not already exist.

    Returns:
        The path of the file in its destination.

    Raises:
        OSError: If the copied file is missing or its size differs from the source.
    """
    copy2(
        src=source_path,
        dst=destination_path
    )

    if not destination_path.is_file() or destination_path.stat().st_size != source_path.stat().st_size:
        destination_path.unlink(missing_ok=True)
        raise OSError(
            f"Copy could not be confirmed, source left in place: "
            f"{source_path} -> {destination_path}"
        )

    source_path.unlink()
    return destination_path