from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image
from loguru import logger

from config.globals import AcceptedImageFormats
from config.paths import CORRUPT_DIR, RAW_DIR, LOGS_DIR
from monitoring.logging_manager import LoggingManager

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


def is_pixel_valid(
        arr: np.ndarray,
        path: Path
) -> bool:
    """
    Check whether a decoded image array contains meaningful pixel data.

    Intended to be called immediately after PIL decode — the caller opens
    the image and converts to a float32 RGB array, then passes it here.
    No IO is performed by this function.

    Catches four degenerate conditions that structural checks miss:
        1. Empty array after decode.
        2. Non-finite values (NaN or Inf) after float conversion.
        3. Zero variance — solid colour, all-black, all-white.
        4. Suspiciously low dynamic range (< 10 on 0-255 scale).

    Args:
        arr: Float32 numpy array of shape (H, W, 3), already decoded by caller.
        path: Path to the source file, used only for logging.

    Returns:
        True if pixel data appears valid for training. False otherwise.
    """
    if arr.size == 0:
        logger.warning(f"Empty array after decode: {path.name}")
        return False

    if not np.isfinite(arr).all():
        logger.warning(f"Non-finite pixel values (NaN/Inf): {path.name}")
        return False

    pixel_range: float = float(arr.max() - arr.min())

    if pixel_range == 0:
        logger.warning(f"Zero variance (solid colour): {path.name}")
        return False

    if pixel_range < 10:
        logger.warning(f"Suspiciously low dynamic range ({pixel_range:.1f}): {path.name}")
        return False

    return True


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
    extensions: frozenset[str] = AcceptedImageFormats.as_frozen_set(include_dot=True)
    candidates: list[Path] = [p for p in root.rglob("*") if p.suffix.lower() in extensions]

    logger.info(f"Scanning {len(candidates)} images under {root}")

    for path in candidates:
        try:
            with Image.open(path) as img:
                arr = np.array(img.convert(mode="RGB"), dtype=np.float32)

            if not is_pixel_valid(arr=arr, path=path):
                bad.append(path)

        except Exception as exc:
            logger.warning(f"Corrupt (structural): {path}  ({exc})")
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
    # start logging sink
    LoggingManager(
        log_dir=LOGS_DIR,
        session_name="quarantine_script"
    )
    # then run the code
    main()
