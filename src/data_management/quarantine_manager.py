import csv
from dataclasses import dataclass, fields
from pathlib import Path

from loguru import logger

from common.constants.class_labels import ClassLabels
from common.constants.source_labels import SourceLabels
from common.utils.safe_move_file import safe_move_file
from data_management.validate_image_manager import ImageCheckResult


@dataclass(frozen=False)
class QuarantineRow:
    moved_status: bool
    label: ClassLabels
    source: SourceLabels
    filename: str
    src_path: Path
    dst_path: Path
    reason: str

    @classmethod
    def columns_names(cls) -> list[str]:
        """Returns the dataclass field names, for use as the CSV file's column names.

        Returns:
            The field names in declaration order.
        """
        return [f.name for f in fields(cls)]

    def to_write_row(self) -> dict[str, str]:
        """Converts the row into a dict of strings ready for csv.DictWriter.writerow().

        Returns:
            A mapping of field name to its string representation, keyed in the same order as columns_names().
        """
        return {f.name: str(getattr(self, f.name)) for f in fields(self)}

    @classmethod
    def from_csv_row(cls, row: dict[str, str]) -> "QuarantineRow":
        """Rebuilds a QuarantineRow from a csv.DictReader row, restoring the original field types.

        Args:
            row: A single row as produced by csv.DictReader.

        Returns:
            A QuarantineRow with bool, enum and Path fields converted back from strings.

        Raises:
            ValueError: If moved_status is not "True" or "False", or label or source is not a valid enum value.
        """
        moved = row["moved_status"]
        if moved not in ("True", "False"):
            raise ValueError(f"Invalid moved_status value: {moved!r}")
        return cls(
            moved_status=moved == "True",
            label=ClassLabels(row["label"]),
            source=SourceLabels(row["source"]),
            filename=row["filename"],
            src_path=Path(row["src_path"]),
            dst_path=Path(row["dst_path"]),
            reason=row["reason"],
        )

# TODO: filename for the csv should be based somewhat on the SessionID class or a unique name.
#  hardcoding the name will result in overwrites.
class QuarantineManager:
    def __init__(
            self,
            failed_image_checks: frozenset[ImageCheckResult],
            quarantine_dir: Path,
    ) -> None:
        """
        Move files that failed the image checks into a quarantine directory and report on them.

        Args:
            failed_image_checks: Results for every file to quarantine (failed targets and non targets).
            quarantine_dir: Directory the files are moved into. The CSV report is saved here too.
        """
        self._failed_image_checks: frozenset[ImageCheckResult] = failed_image_checks
        self._quarantine_dir: Path = quarantine_dir

        # sorted so the report order is stable between runs (a frozenset has no order)
        self._rows: list[QuarantineRow] = [
            QuarantineRow(
                moved_status=False,
                label=failed_check.file_ref.label,
                source=failed_check.file_ref.source,
                filename=failed_check.file_ref.filename,
                src_path=failed_check.file_ref.resolved_filepath,
                dst_path=self._quarantine_dir / failed_check.file_ref.resolved_filepath.name,
                reason=failed_check.join_failed_reasons,
            )
            for failed_check in sorted(self._failed_image_checks, key=lambda check: check.file_ref.filename)
        ]

    def _write_report_csv(self, csv_filepath: Path) -> None:
        """
        Write the current rows to the CSV, swapping the file in only once it is fully written.

        Args:
            csv_filepath: Where the report should end up.
        """
        tmp_filepath = csv_filepath.with_suffix(".csv.tmp")
        with tmp_filepath.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=QuarantineRow.columns_names())
            writer.writeheader()
            for row in self._rows:
                writer.writerow(row.to_write_row())
        tmp_filepath.replace(csv_filepath)  # atomic, so a crash never leaves a half written report

    def gen_quarantine_report_csv(self, quarantine_csv_filename: str) -> Path:
        """
        Save the quarantine report with the current moved_status of every row.

        Args:
            quarantine_csv_filename: Filename of the report, saved inside the quarantine directory.

        Returns:
            Path of the saved CSV.
        """
        self._quarantine_dir.mkdir(parents=True, exist_ok=True)
        csv_filepath = self._quarantine_dir / quarantine_csv_filename
        self._write_report_csv(csv_filepath=csv_filepath)
        return csv_filepath

    def safe_move_failed_image_checks(self) -> None:
        """
        Move each row's file into quarantine and set moved_status=True on the rows that succeed.

        A file that fails to move is logged and left as moved_status=False with the cause added to its reason.
        """
        for row in self._rows:
            try:
                row.dst_path = safe_move_file(
                    source_path=row.src_path,
                    destination_path=row.dst_path,
                    increment_same_filename=True
                )

            except Exception as e:
                logger.error(f"Couldn't quarantine {row.src_path} -> {row.dst_path}, cause={e}")
                row.reason = f"{row.reason}; move failed: {e}"
                continue

            if row.dst_path.is_file() and not row.src_path.exists():
                row.moved_status = True
            else:
                logger.error(f"Move reported success but files don't match: {row.src_path} -> {row.dst_path}")
                row.reason = f"{row.reason}; move unverified"

    def run_quarantine(self, quarantine_csv_filename: str) -> Path:
        """Report, move and re-report the quarantined files.

        Args:
            quarantine_csv_filename: Filename of the report, saved inside the quarantine directory.

        Returns:
            Path of the final CSV report.
        """
        # 1. generate and save the report with moved=False
        quarantine_report_csv_path: Path = self.gen_quarantine_report_csv(
            quarantine_csv_filename=quarantine_csv_filename
        )

        # 2 and 3. move each file and bump its row; always save the report, even if something unexpected stops the loop
        try:
            self.safe_move_failed_image_checks()

        finally:
            self._write_report_csv(csv_filepath=quarantine_report_csv_path)

        num_moved = sum(row.moved_status for row in self._rows)
        logger.info(f"Quarantined {num_moved}/{len(self._rows)} files, report at {quarantine_report_csv_path}")
        return quarantine_report_csv_path
