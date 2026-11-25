from pathlib import Path

from loguru import logger


def rename_file_with_lowercase_extension(
        file_path: Path,
        dry_run: bool = False
) -> None:
    """
    This utility function is used to rename files with lower-case extensions.

    Args:
        file_path: Image file whose extension should be lower-cased.
        dry_run: If True, only log what would be renamed.

    Raises:
        FileExistsError: If a different file with the lower-cased name already exists.
    """
    lowered_path: Path = file_path.with_suffix(suffix=file_path.suffix.lower())
    if lowered_path.name == file_path.name:
        return

    # On a case-insensitive filesystem (macOS default) the lowered name resolves to this same file, which is fine.
    if lowered_path.exists() and not lowered_path.samefile(file_path):
        raise FileExistsError(f"Cannot lower-case extension, a different file already exists: {lowered_path}")

    logger.info(f"{'Would rename' if dry_run else 'Renaming'}: {file_path.name} -> {lowered_path.name}")
    if not dry_run:
        file_path.rename(lowered_path)