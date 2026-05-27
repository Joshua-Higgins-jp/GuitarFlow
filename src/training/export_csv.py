from csv import DictWriter
from pathlib import Path

from loguru import logger

from data_mgmt.image_record_model import ImageRecordModel


def export_datasets_to_csv(
    split_records: dict[str, list[ImageRecordModel]],
    run_timestamp: str,
    output_dir: Path,
) -> None:
    """
    Write one CSV per split recording exactly which images were used in this run.

    image_hash is the canonical record identity — it joins directly back to the
    SQLite database without any path inference. Source and label come from the
    record itself, never inferred from directory names.

    Output filenames are prefixed with the run timestamp so files sort
    chronologically and runs never overwrite each other:
        260522_213000__train.csv
        260522_213000__val.csv
        260522_213000__test.csv

    Columns:
        image_hash — SHA-256 digest; primary key in image_metadata table
        filename   — original filename (e.g. abc123.jpg)
        label      — class name (acoustic / electric / not_guitar)
        source     — image source (unsplash / pixabay / reddit / etc.)
        split      — which split this row belongs to (train / val / test)

    Args:
        split_records: Dict mapping split name → list of ImageRecordModel.
        run_timestamp: Timestamp string formatted.
        output_dir:    Directory to write CSVs into. Created if absent.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    for split_name, records in split_records.items():
        filepath: Path = output_dir / f"{run_timestamp}__{split_name}.csv"

        with open(file=filepath, mode="w", newline="", encoding="utf-8") as file:
            writer = DictWriter(
                f=file,
                fieldnames=[
                    "image_hash",
                    "filename",
                    "label",
                    "source",
                    "split"
                ],
            )
            writer.writeheader()
            for record in records:
                writer.writerow(
                    rowdict={
                        "image_hash": record.image_hash,
                        "filename":   record.filename,
                        "label":      record.label.value,
                        "source":     record.source.value,
                        "split":      split_name}
                )

        logger.info(f"CSV written → {filepath.name}  ({len(records)} rows)")
