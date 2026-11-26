from pathlib import Path
from shutil import copyfileobj, copystat

from loguru import logger


def _exclusive_copy(source_path: Path, destination_path: Path) -> None:
    """Copies a file with its metadata, creating the destination exclusively.

    Args:
        source_path: File to copy.
        destination_path: Where to copy it. Must not already exist.

    Raises:
        FileExistsError: If destination_path already exists. Nothing is created or removed in this case.
    """
    created = False
    try:
        # "xb" = exclusive create: atomically fails if the file exists
        with source_path.open("rb") as src_f, destination_path.open("xb") as dst_f:
            created = True
            copyfileobj(src_f, dst_f)
        copystat(source_path, destination_path)
    except BaseException:
        if created:  # only clean up a partial file we made, never one that was already there
            destination_path.unlink(missing_ok=True)
        raise


def safe_move_file(
        source_path: Path,
        destination_path: Path,
        increment_same_filename: bool = False
) -> Path:
    """
    Moves a file by copying it, confirming the copy, and only then deleting the source.

    An existing file is never overwritten. If the copy does not check out, the bad copy is removed and the source is
    left untouched.

    Args:
        source_path: File to move.
        destination_path: Where it should end up.
        increment_same_filename: If destination_path already exists, save as name_1.ext, name_2.ext, and so on, using
            the first free name, instead of raising.

    Returns:
        The path the file was actually saved to. This differs from destination_path if it was incremented.

    Raises:
        FileExistsError: If destination_path already exists and increment_same_filename is False.
        OSError: If the copied file is missing or its size differs from the source.
    """
    stem, suffix = destination_path.stem, destination_path.suffix
    final_path = destination_path
    counter = 1
    while True:
        try:
            _exclusive_copy(source_path=source_path, destination_path=final_path)
            break
        except FileExistsError:
            if not increment_same_filename:
                raise
            final_path = destination_path.with_name(f"{stem}_{counter}{suffix}")
            counter += 1

    if final_path != destination_path:
        logger.warning(f"{destination_path} already existed, saved as {final_path.name} instead")

    if not final_path.is_file() or final_path.stat().st_size != source_path.stat().st_size:
        final_path.unlink(missing_ok=True)
        raise OSError(f"Copy could not be confirmed, source left in place: {source_path} -> {final_path}")

    source_path.unlink()
    return final_path
