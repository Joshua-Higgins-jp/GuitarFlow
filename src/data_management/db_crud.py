import sqlite3
from dataclasses import dataclass

from loguru import logger

from common.config import DB_TABLE_NAME
from common.constants.status_labels import StatusLabels
from data_management.db_manager import DatabaseManager
from data_management.db_spec import COLUMN_NAMES, REFRESHED_COLUMNS
from data_management.image_record_model import ImageRecordModel

_INSERT_SQL: str = (
    f"INSERT INTO {DB_TABLE_NAME} ({', '.join(COLUMN_NAMES)}) "
    f"VALUES ({', '.join(f':{name}' for name in COLUMN_NAMES)})"
)

# Only the refreshed columns, so pixel_hash, width, height and ingested_at keep the values from the first insert
_UPDATE_SQL: str = (
    f"UPDATE {DB_TABLE_NAME} "
    f"SET {', '.join(f'{name} = :{name}' for name in REFRESHED_COLUMNS)} "
    f"WHERE pixel_hash = :pixel_hash"
)


@dataclass(frozen=True)
class SyncCounts:
    """
    What one scan changed in the image records table.

    Attributes:
        inserted: Records for images seen for the first time.
        updated: Records for images already in the table, now ACTIVE with refreshed fields.
        missing: Rows left MISSING after the scan, including rows that were already MISSING before it.
    """
    inserted: int
    updated: int
    missing: int


class DatabaseCRUD:
    """
    Writes image records to the database, building on the connection made in the manager.

    sync_scan is the only method a scan needs to call. The other methods are the steps it runs, and each takes an open
    connection so that sync_scan can run them all in one transaction.
    """
    def __init__(
            self,
            db_manager: DatabaseManager
    ) -> None:
        """
        Store the database handle without touching the database.

        Args:
            db_manager: A DatabaseManager whose schema has already been verified.

        Raises:
            RuntimeError: If the ImageRecordModel fields and the table columns differ.
        """
        self._db: DatabaseManager = db_manager
        self._check_model_matches_schema()

    @staticmethod
    def _check_model_matches_schema() -> None:
        """
        Fail early if ImageRecordModel and the table columns have drifted apart.

        Raises:
            RuntimeError: If a field exists only in the model or a column exists only in the table.
        """
        model_fields = set(ImageRecordModel.model_fields)
        columns = set(COLUMN_NAMES)
        if model_fields != columns:
            raise RuntimeError(
                f"ImageRecordModel and the '{DB_TABLE_NAME}' schema disagree: "
                f"only in the model {sorted(model_fields - columns)}, only in the table {sorted(columns - model_fields)}"
            )

    def sync_scan(self, records: list[ImageRecordModel]) -> SyncCounts:
        """
        Write the records from one scan to the table in a single transaction.

        First every ACTIVE row is marked MISSING. Then each record is inserted if its hash is new, or its row is
        updated if not, which also makes it ACTIVE again. Rows for images the scan didn't find stay MISSING.

        Marking everything MISSING first keeps the unique index on active paths out of the way: if two files swap
        names, or a new image takes the name of one that was removed, the old row is no longer ACTIVE when the new
        one claims its path. Everything runs in one transaction, so a crash partway through leaves the table as it
        was before the scan.

        Args:
            records: One record per image that passed this scan, all with status ACTIVE.

        Returns:
            How many rows were inserted, updated and left MISSING.
        """
        with self._db.connection() as conn:
            existing_hashes = self.fetch_all_pixel_hashes(conn=conn)
            self.mark_all_missing(conn=conn)

            num_inserted = 0
            num_updated = 0
            for record in records:
                if record.pixel_hash in existing_hashes:
                    self.update_record(conn=conn, record=record)
                    num_updated += 1
                else:
                    self.insert_record(conn=conn, record=record)
                    num_inserted += 1

            num_missing: int = conn.execute(
                f"SELECT COUNT(*) FROM {DB_TABLE_NAME} WHERE status = ?",
                (StatusLabels.MISSING.value,)
            ).fetchone()[0]

        counts = SyncCounts(inserted=num_inserted, updated=num_updated, missing=num_missing)
        logger.info(
            f"Synced '{DB_TABLE_NAME}': inserted={counts.inserted}; updated={counts.updated}; missing={counts.missing}"
        )
        return counts

    @staticmethod
    def fetch_all_pixel_hashes(conn: sqlite3.Connection) -> set[str]:
        """
        Read the pixel hash of every row, whatever its status.

        Args:
            conn: An open connection.

        Returns:
            Every pixel hash in the table.
        """
        rows = conn.execute(f"SELECT pixel_hash FROM {DB_TABLE_NAME}").fetchall()
        return {row["pixel_hash"] for row in rows}

    @staticmethod
    def mark_all_missing(conn: sqlite3.Connection) -> int:
        """
        Set every ACTIVE row to MISSING, ready for a scan to make the images it finds ACTIVE again.

        Args:
            conn: An open connection.

        Returns:
            How many rows were ACTIVE before the call.
        """
        cursor = conn.execute(
            f"UPDATE {DB_TABLE_NAME} SET status = ? WHERE status = ?",
            (StatusLabels.MISSING.value, StatusLabels.ACTIVE.value)
        )
        return cursor.rowcount

    @staticmethod
    def insert_record(conn: sqlite3.Connection, record: ImageRecordModel) -> None:
        """
        Insert a record for an image that is not in the table yet.

        Args:
            conn: An open connection.
            record: The record to insert.

        Raises:
            sqlite3.IntegrityError: If the pixel hash already exists, another ACTIVE row has the same path, or a
                value breaks a CHECK constraint.
        """
        conn.execute(_INSERT_SQL, record.model_dump(mode="json"))

    @staticmethod
    def update_record(conn: sqlite3.Connection, record: ImageRecordModel) -> None:
        """
        Refresh the row of an image that is already in the table, logging any change to its path.

        Only the refreshed columns are written. pixel_hash, width, height and ingested_at keep their original values.
        A move to another label or source changes what the model learns from the image, so it is logged as a
        warning. A rename within the same folder is logged as info.

        Args:
            conn: An open connection.
            record: The record holding the new values.

        Raises:
            KeyError: If no row has the record's pixel hash.
            sqlite3.IntegrityError: If another ACTIVE row has the same path, or a value breaks a CHECK constraint.
        """
        old = conn.execute(
            f"SELECT label, source, filename FROM {DB_TABLE_NAME} WHERE pixel_hash = ?",
            (record.pixel_hash,)
        ).fetchone()
        if old is None:
            raise KeyError(f"No row with pixel_hash {record.pixel_hash}")

        new_row = record.model_dump(mode="json")
        old_path = f"{old['label']}/{old['source']}/{old['filename']}"
        new_path = f"{new_row['label']}/{new_row['source']}/{new_row['filename']}"

        if (old["label"], old["source"]) != (new_row["label"], new_row["source"]):
            logger.warning(f"Image {record.pixel_hash[:12]} moved from {old_path} to {new_path}")
        elif old["filename"] != new_row["filename"]:
            logger.info(f"Image {record.pixel_hash[:12]} renamed from {old_path} to {new_path}")

        conn.execute(_UPDATE_SQL, new_row)
