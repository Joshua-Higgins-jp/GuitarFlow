from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from typing import Optional

from PIL import Image
from loguru import logger

from config.paths import CORRUPT_DIR, RAW_DIR


"""
TO RUN:

    1. cd GuitarFlow/scripts
    2. uv run python quarantine_damaged_images.py --dry-run  OR  
       uv run python quarantine_damaged_images.py --move
    
"""


# ── helpers ──────────────────────────────────────────────────────────────────

def _build_quarantine_name(path: Path, root: Path) -> str:
    """
    Build a quarantine filename that encodes the file's lineage.

    The relative path parts between root and the filename are joined with
    underscores and appended to the original stem, separated by a double
    underscore so the boundary is unambiguous.

    Example
    -------
    root  = /data/raw
    path  = /data/raw/dogs/flickr/image001.jpg
    parts = ['dogs', 'flickr']
    →       image001__dogs_flickr.jpg

    Args:
        path: Absolute path to the image file.
        root: The scan root that was passed to the scanner.

    Returns:
        A flat filename string safe to write into CORRUPT_DIR.
    """
    try:
        relative = path.relative_to(root)

    except ValueError:
        # path is not under root — keep the original name as a fallback
        logger.warning(f"Could not make {path} relative to {root}; using original filename.")
        return path.name

    # parts[:-1] strips the filename itself, leaving only directory parts
    dir_parts = relative.parts[:-1]
    if dir_parts:
        suffix = "_".join(dir_parts)
        return f"{path.stem}__{suffix}{path.suffix}"

    # file sits directly in root — no lineage to append
    return path.name


# ── core logic ────────────────────────────────────────────────────────────────

def find_truncated_and_quarantine(
        root: Path,
        corrupt_dir: Path,
        quarantine: bool = False,  # ← renamed from delete
) -> list[Path]:
    """
    Walk root recursively and attempt to fully load every image.

    PIL's Image.open() is lazy — it does not read pixel data until load()
    is called or an operation requires it. Converting to RGB forces a full
    pixel decode, matching what a training pipeline does.

    Corrupt files are optionally moved to corrupt_dir and renamed to
    preserve their class/source lineage:

        original_stem__class_source.ext

    Args:
        root:        Directory to scan recursively.
        corrupt_dir: Destination directory for quarantined files.
        quarantine:  If True, move corrupt files to corrupt_dir.

    Returns:
        List of paths that failed verification.
    """
    bad: list[Path] = []
    extensions = {".jpg", ".jpeg", ".png", ".webp"}
    candidates = [p for p in root.rglob("*") if p.suffix.lower() in extensions]

    logger.info(f"Scanning {len(candidates)} images under {root}")

    for path in candidates:
        try:
            with Image.open(path) as img:
                img.convert(mode="RGB")
        except Exception as exc:
            logger.warning(f"Corrupt: {path}  ({exc})")
            bad.append(path)

    logger.info(f"Scan complete — {len(bad)} corrupt / {len(candidates)} total")

    if bad and quarantine:
        corrupt_dir.mkdir(parents=True, exist_ok=True)
        for path in bad:
            dest = corrupt_dir / _build_quarantine_name(path, root)
            shutil.move(str(path), dest)
            logger.info(f"Quarantined: {path.name} → {dest.name}")

    return bad


# ── CLI ───────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="quarantine",
        description=(
            "Scan RAW_DIR for corrupt images.\n\n"
            "You must explicitly choose a mode:\n"
            "  --dry-run   log corrupt files only, nothing is moved\n"
            "  --move      log corrupt files AND move them to CORRUPT_DIR\n\n"
            "Quarantined files are renamed to encode their original lineage:\n"
            "  original_stem__class_source.ext"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--dry-run",
        action="store_true",
        help="Log corrupt files only. Nothing is moved.",
    )
    mode.add_argument(
        "--move",
        action="store_true",
        help=f"Move corrupt files to {CORRUPT_DIR}.",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> None:
    """
    Entry point for the quarantine CLI.

    Args:
        argv: Argument list for testing; defaults to sys.argv when None.
    """
    parser = _build_parser()
    args = parser.parse_args(argv)

    bad = find_truncated_and_quarantine(
        root=RAW_DIR,
        corrupt_dir=CORRUPT_DIR,
        quarantine=args.move,
    )
    sys.exit(0 if not bad else 1)


if __name__ == "__main__":
    main()
