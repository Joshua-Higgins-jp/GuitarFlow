import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Generator

from loguru import logger

from src.data_collecting.collectors.collector_models_enums import ImageMetadata, ImageSource, ImageLabel


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
        logger.info(f"Initialised metadata database at {db_path}")

    @contextmanager
    def _get_connection(self) -> Generator[sqlite3.Connection, None, None]:
        """Context manager for database connections with row factory."""
        conn: sqlite3.Connection = sqlite3.connect(database=self.db_path)
        conn.row_factory = sqlite3.Row
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
        with self._get_connection() as conn:  # pycharm is telling me "self parameter unfilled"
            conn.execute(
                """CREATE TABLE IF NOT EXISTS image_metadata (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source TEXT NOT NULL,
                    label TEXT NOT NULL,
                    image_id TEXT NOT NULL UNIQUE,
                    image_url TEXT NOT NULL,
                    search_query TEXT NOT NULL,
                    searched_at TIMESTAMP NOT NULL,
                    license TEXT DEFAULT NULL,
                    width INTEGER DEFAULT NULL,
                    height INTEGER DEFAULT NULL,
                    filesize INTEGER DEFAULT NULL,
                    is_downloaded BOOLEAN DEFAULT 0,
                    downloaded_at TIMESTAMP DEFAULT NULL
                )
            """
        )
            # Create indices for common queries
            conn.execute("""
                         CREATE INDEX IF NOT EXISTS idx_source
                             ON image_metadata(source)
                         """)
            conn.execute("""
                         CREATE INDEX IF NOT EXISTS idx_label
                             ON image_metadata(label)
                         """)
            conn.execute("""
                         CREATE INDEX IF NOT EXISTS idx_search_query
                             ON image_metadata(search_query)
                         """)
            conn.execute("""
                         CREATE INDEX IF NOT EXISTS idx_is_downloaded
                             ON image_metadata(is_downloaded)
                         """)

    def insert(self, metadata: ImageMetadata) -> int:
        """
        Insert image metadata into the database.

        Args:
            metadata: ImageMetadata object to store

        Returns:
            The row ID of the inserted record

        Raises:
            sqlite3.IntegrityError: If image_id already exists
        """
        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                INSERT INTO image_metadata (
                    source, 
                    label, 
                    image_id, 
                    image_url, 
                    search_query,
                    searched_at, 
                    license, 
                    width, 
                    height, 
                    filesize,
                    is_downloaded, 
                    downloaded_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    metadata.source.value,
                    metadata.label.value,
                    metadata.image_id,
                    metadata.image_url,
                    metadata.search_query,
                    metadata.searched_at.isoformat(),
                    metadata.license,
                    metadata.width,
                    metadata.height,
                    metadata.filesize,
                    metadata.is_downloaded,
                    metadata.downloaded_at.isoformat() if metadata.downloaded_at else None
                )
            )
            row_id = cursor.lastrowid
            logger.debug(f"Inserted image {metadata.image_id} with row_id={row_id}")
            return row_id

    def get_all(
            self,
            source: Optional[ImageSource] = None,
            label: Optional[ImageLabel] = None
    ) -> List[ImageMetadata]:
        """
        Retrieve all metadata, optionally filtered by source and/or label.

        Args:
            source: Filter by image source (optional)
            label: Filter by image label (optional)

        Returns:
            List of ImageMetadata objects
        """
        query = "SELECT * FROM image_metadata WHERE 1=1"
        params = []

        if source is not None:
            query += " AND source = ?"
            params.append(source.value)

        if label is not None:
            query += " AND label = ?"
            params.append(label.value)

        query += " ORDER BY downloaded_at DESC"

        with self._get_connection() as conn:  # pycharm is telling me "self parameter unfilled"
            rows = conn.execute(query, params).fetchall()
            return [self._row_to_metadata(row) for row in rows]

    @staticmethod
    def _row_to_metadata(row: sqlite3.Row) -> ImageMetadata:
        """
        Convert a database row to an ImageMetadata object.

        Args:
            row: SQLite row object

        Returns:
            ImageMetadata instance
        """
        return ImageMetadata(
            source=ImageSource(row['source']),
            label=ImageLabel(row['label']),
            image_id=row['image_id'],
            image_url=row['image_url'],
            search_query=row['search_query'],
            searched_at=datetime.fromisoformat(row['searched_at']),
            license=row['license'],
            width=row['width'],
            height=row['height'],
            filesize=row['filesize'],
            is_downloaded=bool(row['is_downloaded']),
            downloaded_at=datetime.fromisoformat(row['downloaded_at']) if row['downloaded_at'] else None,
        )

    def get_by_id(self, image_id: str) -> Optional[ImageMetadata]:
        """
        Retrieve a specific image metadata record by image_id.

        Args:
            image_id: The unique image identifier

        Returns:
            ImageMetadata object if found, None otherwise
        """
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM image_metadata WHERE image_id = ?",
                (image_id,)
            ).fetchone()

            if row is None:
                return None

            return self._row_to_metadata(row)

    def update_download_status(
            self,
            image_id: str,
            width: Optional[int] = None,
            height: Optional[int] = None,
            filesize: Optional[int] = None
    ) -> bool:
        """
        Mark an image as downloaded and optionally update its dimensions and filesize.

        Args:
            image_id: The unique image identifier
            width: Image width in pixels (optional)
            height: Image height in pixels (optional)
            filesize: File size in bytes (optional)

        Returns:
            True if the record was updated, False if image_id not found
        """
        with self._get_connection() as conn:  # pycharm is telling me "self parameter unfilled"
            cursor = conn.execute(
                """
                UPDATE image_metadata
                SET is_downloaded = 1,
                    downloaded_at = ?,
                    width         = COALESCE(?, width),
                    height        = COALESCE(?, height),
                    filesize      = COALESCE(?, filesize)
                WHERE image_id = ?
                """,
                (
                    datetime.now().isoformat(),
                    width,
                    height,
                    filesize,
                    image_id
                )
            )

            updated = cursor.rowcount > 0
            if updated:
                logger.debug(f"Marked image {image_id} as downloaded")
            else:
                logger.warning(f"Image {image_id} not found for update")

            return updated

    def delete(self, image_id: str) -> bool:
        """
        Delete a specific image metadata record.

        Args:
            image_id: The unique image identifier to delete

        Returns:
            True if a record was deleted, False if image_id not found
        """
        with self._get_connection() as conn:
            cursor = conn.execute(
                "DELETE FROM image_metadata WHERE image_id = ?",
                (image_id,)
            )

            deleted = cursor.rowcount > 0
            if deleted:
                logger.info(f"Deleted image {image_id}")
            else:
                logger.warning(f"Image {image_id} not found for deletion")

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
                            source, 
                            label, 
                            image_id, 
                            image_url, 
                            search_query,
                            searched_at, 
                            license, 
                            width, 
                            height, 
                            filesize, 
                            is_downloaded, 
                            downloaded_at
                        )
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            metadata.source.value,
                            metadata.label.value,
                            metadata.image_id,
                            metadata.image_url,
                            metadata.search_query,
                            metadata.searched_at.isoformat(),
                            metadata.license,
                            metadata.width,
                            metadata.height,
                            metadata.filesize,
                            metadata.is_downloaded,
                            metadata.downloaded_at.isoformat() if metadata.downloaded_at else None
                        )
                    )
                    stats['inserted'] += 1
                except sqlite3.IntegrityError:
                    logger.debug(f"Image {metadata.image_id} already exists, skipping")
                    stats['skipped'] += 1

        logger.info(f"Batch insert: {stats['inserted']} inserted, {stats['skipped']} skipped")
        return stats
