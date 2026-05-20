from contextlib import contextmanager
from datetime import datetime
from enum import Enum
from pathlib import Path
from sqlite3 import Connection, connect, IntegrityError, Row
from typing import Any, Dict, Iterator, List, Optional, Set

from loguru import logger

from config.globals import (
    ClassLabels,
    SourceLabels,
    StatusLabels,
)
from data_mgmt.image_record_model import ImageRecordModel

# ---------------------------------------------------------------------------
# Column ordering — single source of truth for INSERT / SELECT mapping.
# Must match _init_schema and ImageRecordModel fields exactly.
# ---------------------------------------------------------------------------
_COLUMNS: tuple[str, ...] = (
    "image_hash",
    "source",
    "label",
    "status",
    "source_id",
    "image_url",
    "filename",
    "search_query",
    "acquired_at",
    "ingested_at",
    "last_seen",
    "image_license",
    "width",
    "height",
    "filesize_bytes",
    "image_format",
    "num_channels",
)


class ImageRecordAdapter:
    """
    Translates between ImageRecordModel instances and SQLite rows.

    ImageRecordModel is a pure data and validation object with no knowledge
    of SQLite. This adapter owns that boundary — it is the only place in the
    codebase that knows about both the model and the database row format.

    All methods are static. This class is never instantiated; it is a
    namespace for related conversion logic, not a stateful object.

    Column ordering is governed by the module-level _COLUMNS constant,
    which must remain in sync with the SQLite schema and the model fields.
    """
    # @staticmethod
    # def columns() -> tuple[str, ...]:
    #     """
    #     Return the canonical column ordering for INSERT and SELECT operations.
    #
    #     Returns:
    #         Tuple of column name strings matching the SQLite schema.
    #     """
    #     return _COLUMNS

    @staticmethod
    def insert_sql(table: str = "image_metadata") -> str:
        """
        Generate a parameterised INSERT statement for the image_metadata table.

        Args:
            table: Target table name. Defaults to image_metadata.

        Returns:
            SQL string with ? placeholders in _COLUMNS order.
        """
        return (
            f"INSERT INTO {table} ({', '.join(_COLUMNS)}) "
            f"VALUES ({', '.join('?' for _ in _COLUMNS)})"
        )

    @staticmethod
    def record_to_row(record: ImageRecordModel) -> tuple:
        """
        Serialise an ImageRecordModel to an ordered tuple for SQLite parameter binding.

        Converts enums to their .value and datetimes to .isoformat().
        Column order matches _COLUMNS.

        Args:
            record: The model instance to serialise.

        Returns:
            Tuple of SQLite-compatible values in _COLUMNS order.
        """
        def _serialise(value: Any) -> Any:
            if isinstance(value, Enum):
                return value.value

            if isinstance(value, datetime):
                return value.isoformat()

            return value

        return tuple(_serialise(getattr(record, col)) for col in _COLUMNS)

    @staticmethod
    def row_to_record(row: Row) -> ImageRecordModel:
        """
        Reconstruct an ImageRecordModel from a sqlite3.Row.

        Pydantic v2 coerces raw strings into the correct enum and datetime
        types via field validators, so a plain dict unpack is sufficient.

        Args:
            row: A sqlite3.Row returned from a SELECT query.

        Returns:
            Populated ImageRecordModel instance.
        """
        return ImageRecordModel(**dict(row))


# ---------------------------------------------------------------------------
# Database manager
# ---------------------------------------------------------------------------
class ImageMetadataDBManager:
    """
    SQLite database manager for image metadata.

    image_hash is the primary key — identity is content, not a generated ID.
    The same image file always maps to the same record regardless of when
    it was ingested or whether it was temporarily missing.

    Column mapping is derived from _COLUMNS so that insert, batch insert,
    and row-to-model conversion stay in sync automatically. To add a column:
        1. Add the field to ImageRecordModel
        2. Add the column name to _COLUMNS (same position)
        3. Add the column to _init_schema
        4. Update _row_to_record if the type needs re-hydration
    """
    def __init__(self, db_path: Path) -> None:
        """
        Initialise the metadata database.

        Creates the schema if the database file does not yet exist,
        or validates the connection if it does.

        Args:
            db_path: Path to the SQLite database file.
        """
        if not isinstance(db_path, Path):
            raise TypeError("db_path must be a Path object")

        self.db_path: Path = db_path
        self._init_schema()

    @contextmanager
    def _get_connection(self) -> Iterator[Connection]:
        """Context manager for database connections with row factory."""
        conn: Connection = connect(database=self.db_path)
        conn.row_factory = Row
        try:
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            logger.error(f"Database error: {e}")
            raise
        finally:
            conn.close()

    def _init_schema(self) -> None:
        """Create the image_metadata table if it does not exist."""
        with self._get_connection() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS image_metadata (
                    image_hash      TEXT    PRIMARY KEY,
                    source          TEXT    NOT NULL,
                    label           TEXT    NOT NULL,
                    status          TEXT    NOT NULL DEFAULT 'active',
                    source_id       TEXT    DEFAULT NULL,
                    image_url       TEXT    DEFAULT NULL,
                    filename        TEXT    NOT NULL,
                    search_query    TEXT    DEFAULT NULL,
                    acquired_at     TEXT    NOT NULL,
                    ingested_at     TEXT    NOT NULL,
                    last_seen       TEXT    NOT NULL,
                    image_license   TEXT    DEFAULT NULL,
                    width           INTEGER NOT NULL,
                    height          INTEGER NOT NULL,
                    filesize_bytes  INTEGER NOT NULL,
                    image_format    TEXT    NOT NULL,
                    num_channels    INTEGER NOT NULL
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_source ON image_metadata(source)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_label  ON image_metadata(label)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_status ON image_metadata(status)")

        logger.info(f"Initialised database schema at {self.db_path}")

    # ------------------------------------------------------------------
    # INSERT
    # ------------------------------------------------------------------
    def insert(self, record: ImageRecordModel) -> None:
        """
        Insert a single image metadata record.

        Args:
            record: ImageRecordModel to store.

        Raises:
            sqlite3.IntegrityError: If image_hash already exists.
        """
        with self._get_connection() as conn:
            conn.execute(
                ImageRecordAdapter.insert_sql(),
                ImageRecordAdapter.record_to_row(record=record)
            )
        logger.debug(f"Inserted image {record.image_hash[:12]}...")

    def insert_batch(self, records: List[ImageRecordModel]) -> Dict[str, int]:
        """
        Insert multiple records in one transaction.

        Duplicate hashes are silently skipped via IntegrityError catch.

        Args:
            records: List of ImageRecordModel objects to insert.

        Returns:
            Dict with 'inserted' and 'skipped' counts.
        """
        stats: Dict[str, int] = {"inserted": 0, "skipped": 0}

        with self._get_connection() as conn:
            for record in records:
                try:
                    conn.execute(
                        ImageRecordAdapter.insert_sql(),
                        ImageRecordAdapter.record_to_row(record=record)
                    )
                    stats["inserted"] += 1

                except IntegrityError as e:
                    logger.debug(f"Image {record.image_hash[:12]}... already exists, skipping: {e}")
                    stats["skipped"] += 1

        logger.info(f"Batch insert: {stats['inserted']} inserted, {stats['skipped']} skipped")
        return stats

    # ------------------------------------------------------------------
    # READ
    # ------------------------------------------------------------------
    def get_by_hash(self, image_hash: str) -> Optional[ImageRecordModel]:
        """
        Retrieve a single record by image hash.

        Args:
            image_hash: SHA-256 hex digest.

        Returns:
            ImageRecordModel if found, None otherwise.
        """
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM image_metadata WHERE image_hash = ?",
                (image_hash,)
            ).fetchone()
            return ImageRecordAdapter.row_to_record(row=row) if row else None

    def get_all(
            self,
            source: Optional[SourceLabels] = None,
            label: Optional[ClassLabels] = None,
            status: Optional[StatusLabels] = None,
            limit: Optional[int] = None,
            offset: int = 0,
    ) -> List[ImageRecordModel]:
        """
        Retrieve records with optional filters.

        Args:
            source: Filter by image source.
            label: Filter by class label.
            status: Filter by active/missing status.
            limit: Max records to return.
            offset: Records to skip (pagination).

        Returns:
            List of ImageRecordModel objects.
        """
        query: str = "SELECT * FROM image_metadata"
        params: list = []
        conditions: list[str] = []

        if source is not None:
            conditions.append("source = ?")
            params.append(source.value)

        if label is not None:
            conditions.append("label = ?")
            params.append(label.value)

        if status is not None:
            conditions.append("status = ?")
            params.append(status.value)

        if conditions:
            query += " WHERE " + " AND ".join(conditions)

        query += " ORDER BY last_seen DESC"

        if limit is not None:
            query += " LIMIT ?"
            params.append(limit)

        if offset > 0:
            query += " OFFSET ?"
            params.append(offset)

        with self._get_connection() as conn:
            rows = conn.execute(query, params).fetchall()
            return [ImageRecordAdapter.row_to_record(row=row) for row in rows]

    def get_all_hashes(self) -> Set[str]:
        """
        Return the set of all known image hashes.

        Used by the reconciler to determine which disk files are new
        vs already known (even if currently marked missing).

        Returns:
            Set of SHA-256 hex digest strings.
        """
        with self._get_connection() as conn:
            rows = conn.execute("SELECT image_hash FROM image_metadata").fetchall()
            return {row["image_hash"] for row in rows}

    def count(
            self,
            source: Optional[SourceLabels] = None,
            label: Optional[ClassLabels] = None,
            status: Optional[StatusLabels] = None,
    ) -> int:
        """
        Count records, optionally filtered.

        Args:
            source: Filter by image source.
            label: Filter by class label.
            status: Filter by status.

        Returns:
            Count of matching records.
        """
        query: str = "SELECT COUNT(*) as count FROM image_metadata"
        params: list = []
        conditions: list[str] = []

        if source is not None:
            conditions.append("source = ?")
            params.append(source.value)

        if label is not None:
            conditions.append("label = ?")
            params.append(label.value)

        if status is not None:
            conditions.append("status = ?")
            params.append(status.value)

        if conditions:
            query += " WHERE " + " AND ".join(conditions)

        with self._get_connection() as conn:
            row = conn.execute(query, params).fetchone()
            return row["count"]

    # ------------------------------------------------------------------
    # UPDATE (reconciler operations)
    # ------------------------------------------------------------------
    def mark_missing(self, image_hash: str) -> bool:
        """
        Mark a single image as missing (file no longer found on disk).

        Args:
            image_hash: SHA-256 hex digest.

        Returns:
            True if the record was updated, False if hash not found.
        """
        with self._get_connection() as conn:
            cursor = conn.execute(
                "UPDATE image_metadata SET status = 'missing' WHERE image_hash = ?",
                (image_hash,)
            )
            updated: bool = cursor.rowcount > 0

        if updated:
            logger.info(f"Marked {image_hash[:12]}... as missing")
        else:
            logger.warning(f"mark_missing: hash {image_hash[:12]}... not found in DB")

        return updated

    def confirm_active(self, image_hash: str, last_seen: datetime) -> bool:
        """
        Confirm an image is still present on disk.

        Updates last_seen and ensures status is 'active'. Lightweight
        confirmation used by the reconciler on every run.

        Args:
            image_hash: SHA-256 hex digest.
            last_seen: Timestamp of when the file was confirmed on disk.

        Returns:
            True if the record was updated, False if hash not found.
        """
        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                UPDATE image_metadata
                SET status    = 'active',
                    last_seen = ?
                WHERE image_hash = ?
                """,
                (last_seen.isoformat(), image_hash)
            )
            updated: bool = cursor.rowcount > 0

        if not updated:
            logger.warning(f"confirm_active: hash {image_hash[:12]}... not found in DB")

        return updated

    def bulk_mark_missing(self, image_hashes: Set[str]) -> int:
        """
        Mark multiple images as missing in a single transaction.

        Batches the UPDATE into chunks of 500 to stay within SQLite's
        variable placeholder limit (default 999).

        Args:
            image_hashes: Set of SHA-256 hex digests to mark as missing.

        Returns:
            Total number of rows updated.
        """
        if not image_hashes:
            return 0

        total_updated: int = 0
        hash_list: list[str] = list(image_hashes)
        batch_size: int = 500

        with self._get_connection() as conn:
            for i in range(0, len(hash_list), batch_size):
                batch = hash_list[i:i + batch_size]
                placeholders = ", ".join("?" for _ in batch)
                cursor = conn.execute(
                    f"UPDATE image_metadata SET status = 'missing' "
                    f"WHERE image_hash IN ({placeholders})",
                    batch,
                )
                total_updated += cursor.rowcount

        logger.info(f"bulk_mark_missing: marked {total_updated} records as missing")
        return total_updated

    # ------------------------------------------------------------------
    # DELETE
    # ------------------------------------------------------------------
    def delete(self, image_hash: str) -> bool:
        """
        Permanently delete a metadata record.

        Prefer mark_missing over delete for files that may return.
        Use delete only for confirmed duplicates or data errors.

        Args:
            image_hash: SHA-256 hex digest of the image to delete.

        Returns:
            True if a record was deleted, False if hash not found.
        """
        with self._get_connection() as conn:
            cursor = conn.execute(
                "DELETE FROM image_metadata WHERE image_hash = ?",
                (image_hash,),
            )
            deleted: bool = cursor.rowcount > 0

        if deleted:
            logger.info(f"Deleted image {image_hash[:12]}...")
        else:
            logger.warning(f"delete: hash {image_hash[:12]}... not found")

        return deleted
