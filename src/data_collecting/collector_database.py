from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from sqlite3 import Connection, connect, IntegrityError, Row
from typing import Optional, List, Dict, Iterator
from uuid import UUID

from loguru import logger

from data_collecting.metadata_models_enums import ImageMetadata, ImageSource, ImageLabel


class ImageMetadataDBManager:
    """
    SQLite database manager for image metadata.

    Provides seamless storage and retrieval of ImageMetadata objects.
    """
    def __init__(
            self,
            db_path: Path
    ) -> None:
        """
        Initialize the metadata database.

        Args:
            db_path: Path to the SQLite database file (e.g., Path("guitarflow_metadata.db"))
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
        """Create the image_metadata table if it doesn't exist."""
        with self._get_connection() as conn:
            conn.execute(
                """CREATE TABLE IF NOT EXISTS image_metadata
                   (
                       id                INTEGER PRIMARY KEY AUTOINCREMENT,
                       internal_id       TEXT      NOT NULL UNIQUE,
                       source            TEXT      NOT NULL,
                       label             TEXT      NOT NULL,
                       source_id         TEXT      DEFAULT NULL,
                       image_url         TEXT      DEFAULT NULL,
                       original_filename TEXT      NOT NULL,
                       search_query      TEXT      DEFAULT NULL,
                       acquired_at       TIMESTAMP NOT NULL,
                       ingested_at       TIMESTAMP NOT NULL,
                       license           TEXT      DEFAULT NULL,
                       width             INTEGER   NOT NULL,
                       height            INTEGER   NOT NULL,
                       filesize_bytes    INTEGER   NOT NULL,
                       UNIQUE(source, source_id)
                   )
                """
            )

            # Create indices for common queries
            conn.execute("""
                         CREATE INDEX IF NOT EXISTS idx_internal_id
                             ON image_metadata(internal_id)
                         """)
            conn.execute("""
                         CREATE INDEX IF NOT EXISTS idx_source
                             ON image_metadata(source)
                         """)
            conn.execute("""
                         CREATE INDEX IF NOT EXISTS idx_label
                             ON image_metadata(label)
                         """)
            conn.execute("""
                         CREATE INDEX IF NOT EXISTS idx_source_id
                             ON image_metadata(source_id)
                         """)

        logger.info(f"Successfully initialized database schema at {self.db_path}")

    def insert(self, metadata: ImageMetadata) -> int:
        """
        Insert image metadata into the database.

        Args:
            metadata: ImageMetadata object to store

        Returns:
            The row ID of the inserted record

        Raises:
            sqlite3.IntegrityError: If internal_id already exists or (source, source_id) conflicts
        """
        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                INSERT INTO image_metadata (
                    internal_id,
                    source, 
                    label, 
                    source_id,
                    image_url,
                    original_filename,
                    search_query,
                    acquired_at,
                    ingested_at,
                    license, 
                    width, 
                    height, 
                    filesize_bytes
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(metadata.internal_id),
                    metadata.source.value,
                    metadata.label.value,
                    metadata.source_id,
                    metadata.image_url,
                    metadata.original_filename,
                    metadata.search_query,
                    metadata.acquired_at.isoformat(),
                    metadata.ingested_at.isoformat(),
                    metadata.license,
                    metadata.width,
                    metadata.height,
                    metadata.filesize_bytes
                )
            )
            row_id = cursor.lastrowid
            logger.debug(f"Inserted image {metadata.internal_id} with row_id={row_id}")
            return row_id

    def get_all(
            self,
            source: Optional[ImageSource] = None,
            label: Optional[ImageLabel] = None,
            limit: Optional[int] = None,
            offset: int = 0
    ) -> List[ImageMetadata]:
        """
        Retrieve all metadata, optionally filtered by source and/or label.

        Args:
            source: Filter by image source (optional)
            label: Filter by image label (optional)
            limit: Maximum number of records to return (optional, for pagination)
            offset: Number of records to skip (default: 0, for pagination)

        Returns:
            List of ImageMetadata objects
        """
        query = "SELECT * FROM image_metadata"
        params = []
        conditions = []

        if source is not None:
            conditions.append("source = ?")
            params.append(source.value)

        if label is not None:
            conditions.append("label = ?")
            params.append(label.value)

        if conditions:
            query += " WHERE " + " AND ".join(conditions)

        query += " ORDER BY ingested_at DESC"

        if limit is not None:
            query += " LIMIT ?"
            params.append(limit)

        if offset > 0:
            query += " OFFSET ?"
            params.append(offset)

        with self._get_connection() as conn:
            rows = conn.execute(query, params).fetchall()
            return [self._row_to_metadata(row) for row in rows]

    @staticmethod
    def _row_to_metadata(row: Row) -> ImageMetadata:
        """
        Convert a database row to an ImageMetadata object.

        Args:
            row: SQLite row object

        Returns:
            ImageMetadata instance
        """
        return ImageMetadata(
            internal_id=UUID(row['internal_id']),
            source=ImageSource(row['source']),
            label=ImageLabel(row['label']),
            source_id=row['source_id'],
            image_url=row['image_url'],
            original_filename=row['original_filename'],
            search_query=row['search_query'],
            acquired_at=datetime.fromisoformat(row['acquired_at']),
            ingested_at=datetime.fromisoformat(row['ingested_at']),
            license=row['license'],
            width=row['width'],
            height=row['height'],
            filesize_bytes=row['filesize_bytes']
        )

    def get_by_internal_id(self, internal_id: UUID) -> Optional[ImageMetadata]:
        """
        Retrieve a specific image metadata record by internal_id.

        Args:
            internal_id: The unique internal identifier (UUID)

        Returns:
            ImageMetadata object if found, None otherwise
        """
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM image_metadata WHERE internal_id = ?",
                (str(internal_id),)
            ).fetchone()

            if row is None:
                return None

            return self._row_to_metadata(row)

    def get_by_source_id(
            self,
            source: ImageSource,
            source_id: str
    ) -> Optional[ImageMetadata]:
        """
        Retrieve a specific image metadata record by source and source_id.

        Args:
            source: The image source
            source_id: The source-specific identifier

        Returns:
            ImageMetadata object if found, None otherwise
        """
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM image_metadata WHERE source = ? AND source_id = ?",
                (source.value, source_id)
            ).fetchone()

            if row is None:
                return None

            return self._row_to_metadata(row)

    def delete(self, internal_id: UUID) -> bool:
        """
        Delete a specific image metadata record.

        Args:
            internal_id: The unique internal identifier to delete

        Returns:
            True if a record was deleted, False if internal_id not found
        """
        with self._get_connection() as conn:
            cursor = conn.execute(
                "DELETE FROM image_metadata WHERE internal_id = ?",
                (str(internal_id),)
            )

            deleted = cursor.rowcount > 0
            if deleted:
                logger.info(f"Deleted image {internal_id}")
            else:
                logger.warning(f"Image {internal_id} not found for deletion")

            return deleted

    def insert_batch(self, metadata_list: List[ImageMetadata]) -> Dict[str, int]:
        """
        Insert multiple metadata records in one transaction.

        Args:
            metadata_list: List of ImageMetadata objects to insert

        Returns:
            Dictionary with 'inserted' and 'skipped' counts
        """
        stats = {'inserted': 0, 'skipped': 0}

        with self._get_connection() as conn:
            for metadata in metadata_list:
                try:
                    conn.execute(
                        """
                        INSERT INTO image_metadata (
                            internal_id,
                            source, 
                            label, 
                            source_id,
                            image_url,
                            original_filename,
                            search_query,
                            acquired_at,
                            ingested_at,
                            license, 
                            width, 
                            height, 
                            filesize_bytes
                        )
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            str(metadata.internal_id),
                            metadata.source.value,
                            metadata.label.value,
                            metadata.source_id,
                            metadata.image_url,
                            metadata.original_filename,
                            metadata.search_query,
                            metadata.acquired_at.isoformat(),
                            metadata.ingested_at.isoformat(),
                            metadata.license,
                            metadata.width,
                            metadata.height,
                            metadata.filesize_bytes
                        )
                    )
                    stats['inserted'] += 1
                except IntegrityError as e:
                    logger.debug(f"Image {metadata.internal_id} already exists, skipping: {e}")
                    stats['skipped'] += 1

        logger.info(f"Batch insert: {stats['inserted']} inserted, {stats['skipped']} skipped")
        return stats

    def count(
            self,
            source: Optional[ImageSource] = None,
            label: Optional[ImageLabel] = None
    ) -> int:
        """
        Count total images, optionally filtered by source and/or label.

        Args:
            source: Filter by image source (optional)
            label: Filter by image label (optional)

        Returns:
            Count of matching records
        """
        query = "SELECT COUNT(*) as count FROM image_metadata"
        params = []
        conditions = []

        if source is not None:
            conditions.append("source = ?")
            params.append(source.value)

        if label is not None:
            conditions.append("label = ?")
            params.append(label.value)

        if conditions:
            query += " WHERE " + " AND ".join(conditions)

        with self._get_connection() as conn:
            row = conn.execute(query, params).fetchone()
            return row['count']
