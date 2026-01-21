import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict

from loguru import logger

from src.data_collecting.collectors.collector_models_enums import ImageMetadata, ImageSource, ImageLabel


class ImageMetadataDB:
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
        self.db_path: Path = db_path
        self._init_schema()
        logger.info(f"Initialized metadata database at {db_path}")

    @contextmanager
    def _get_connection(self):
        """Context manager for database connections with row factory."""
        conn = sqlite3.connect(self.db_path)
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
        """Create the images table if it doesn't exist."""
        with self._get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS images (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source TEXT NOT NULL,
                    label TEXT NOT NULL,
                    source_id TEXT NOT NULL UNIQUE,
                    url TEXT NOT NULL,
                    search_query TEXT NOT NULL,
                    license TEXT NOT NULL,
                    downloaded_at TEXT NOT NULL,
                    width INTEGER,
                    height INTEGER,
                    filesize INTEGER,
                    created_at TIMESTAMP,
                    is_trainable BOOLEAN DEFAULT 1
                )
            """)

            # Create indices for common queries
            conn.execute("""
                 CREATE INDEX IF NOT EXISTS idx_source 
                 ON images(source)
             """)
            conn.execute("""
                 CREATE INDEX IF NOT EXISTS idx_label 
                 ON images(label)
             """)
            conn.execute("""
                 CREATE INDEX IF NOT EXISTS idx_search_query 
                 ON images(search_query)
             """)
            conn.execute("""
                 CREATE INDEX IF NOT EXISTS idx_is_trainable 
                 ON images(is_trainable)
             """)

    def insert(self, metadata: ImageMetadata) -> int:
        """
        Insert image metadata into the database.

        Args:
            metadata: ImageMetadata object to store

        Returns:
            The row ID of the inserted record

        Raises:
            sqlite3.IntegrityError: If source_id already exists
        """
        with self._get_connection() as conn:
            cursor = conn.execute("""
                INSERT INTO images (
                    source, label, source_id, url, search_query, 
                    license, downloaded_at, width, height, filesize, is_trainable
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                metadata.source.value,
                metadata.label.value,
                metadata.source_id,
                metadata.url,
                metadata.search_query,
                metadata.license,
                metadata.downloaded_at.isoformat(),
                metadata.width,
                metadata.height,
                metadata.filesize,
                metadata.is_trainable
            ))
            row_id = cursor.lastrowid
            logger.debug(f"Inserted image {metadata.source_id} with row_id={row_id}")
            return row_id

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
                    conn.execute("""
                                 INSERT INTO images (source, label, source_id, url, search_query,
                                                     license, downloaded_at, width, height, filesize, is_trainable)
                                 VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                                 """, (
                                     metadata.source.value,
                                     metadata.label.value,
                                     metadata.source_id,
                                     metadata.url,
                                     metadata.search_query,
                                     metadata.license,
                                     metadata.downloaded_at.isoformat(),
                                     metadata.width,
                                     metadata.height,
                                     metadata.filesize,
                                     metadata.is_trainable
                                 ))
                    stats['inserted'] += 1
                except sqlite3.IntegrityError:
                    logger.debug(f"Image {metadata.source_id} already exists, skipping")
                    stats['skipped'] += 1

        logger.info(f"Batch insert: {stats['inserted']} inserted, {stats['skipped']} skipped")
        return stats

    @staticmethod
    def generate_filename(metadata: ImageMetadata) -> str:
        """
        Generate filename for an image based on its metadata.

        This matches the naming convention used by collectors:
        {source}_{source_id}.jpg

        Args:
            metadata: ImageMetadata object

        Returns:
            Filename string
        """
        return f"{metadata.source.value.lower()}_{metadata.source_id}.jpg"

    def get_by_source_id(self, source_id: str) -> Optional[ImageMetadata]:
        """
        Retrieve metadata by source_id.

        Args:
            source_id: Unique identifier from the image source

        Returns:
            ImageMetadata object if found, None otherwise
        """
        with self._get_connection() as conn:
            row = conn.execute("""
                SELECT * FROM images WHERE source_id = ?
            """, (source_id,)).fetchone()

            if row is None:
                return None

            return self._row_to_metadata(row)

    def get_all(self,
                source: Optional[ImageSource] = None,
                label: Optional[ImageLabel] = None) -> List[ImageMetadata]:
        """
        Retrieve all metadata, optionally filtered by source and/or label.

        Args:
            source: Filter by image source (optional)
            label: Filter by image label (optional)

        Returns:
            List of ImageMetadata objects
        """
        query = "SELECT * FROM images WHERE 1=1"
        params = []

        if source is not None:
            query += " AND source = ?"
            params.append(source.value)

        if label is not None:
            query += " AND label = ?"
            params.append(label.value)

        query += " ORDER BY downloaded_at DESC"

        with self._get_connection() as conn:
            rows = conn.execute(query, params).fetchall()
            return [self._row_to_metadata(row) for row in rows]

    def count(self,
              source: Optional[ImageSource] = None,
              label: Optional[ImageLabel] = None) -> int:
        """
        Count images, optionally filtered by source and/or label.

        Args:
            source: Filter by image source (optional)
            label: Filter by image label (optional)

        Returns:
            Number of matching images
        """
        query = "SELECT COUNT(*) FROM images WHERE 1=1"
        params = []

        if source is not None:
            query += " AND source = ?"
            params.append(source.value)

        if label is not None:
            query += " AND label = ?"
            params.append(label.value)

        with self._get_connection() as conn:
            return conn.execute(query, params).fetchone()[0]

    def exists(self, source_id: str) -> bool:
        """
        Check if an image with the given source_id already exists.

        Args:
            source_id: Unique identifier from the image source

        Returns:
            True if exists, False otherwise
        """
        with self._get_connection() as conn:
            count = conn.execute("""
                SELECT COUNT(*) FROM images WHERE source_id = ?
            """, (source_id,)).fetchone()[0]
            return count > 0

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
            source_id=row['source_id'],
            url=row['url'],
            search_query=row['search_query'],
            license=row['license'],
            downloaded_at=datetime.fromisoformat(row['downloaded_at']),
            width=row['width'],
            height=row['height'],
            filesize=row['filesize'],
            is_trainable=bool(row['is_trainable'])
        )

    def sync_trainable_from_directory(self,
                                      filtered_dir: Path) -> dict[str, int]:
        """
        Synchronize is_trainable status based on filtered directory contents.

        Images present in filtered_dir → is_trainable=True
        Images absent from filtered_dir → is_trainable=False

        This uses the filesystem as the source of truth after manual QC.

        Args:
            filtered_dir: Path to filtered (curated) images directory.
                         Should mirror the structure: filtered/{service}/{label}/

        Returns:
            Dictionary with sync statistics:
            - 'marked_trainable': Number of images marked as trainable
            - 'marked_untrainable': Number of images marked as untrainable
            - 'unchanged': Number of images with unchanged status
        """
        # Recursively get all image files from filtered directory
        filtered_files = {
            f.name
            for f in filtered_dir.rglob('*')
            if f.is_file() and f.suffix.lower() in {'.jpg', '.jpeg', '.png'}
        }

        logger.info(f"Found {len(filtered_files)} files in filtered directory tree")

        stats = {'marked_trainable': 0, 'marked_untrainable': 0, 'unchanged': 0}

        with self._get_connection() as conn:
            # Get all images with their current status
            rows = conn.execute("""
                SELECT source_id, source, is_trainable FROM images
            """).fetchall()

            for row in rows:
                # Reconstruct the filename using same logic as collector
                temp_metadata = ImageMetadata(
                    source=ImageSource(row['source']),
                    label=ImageLabel.ELECTRIC,  # Doesn't matter for filename
                    source_id=row['source_id'],
                    url="",  # Doesn't matter for filename
                    search_query="",
                    license="",
                    downloaded_at=datetime.now()
                )
                expected_filename = self.generate_filename(temp_metadata)

                current_status = bool(row['is_trainable'])
                should_be_trainable = expected_filename in filtered_files

                if current_status == should_be_trainable:
                    stats['unchanged'] += 1
                elif should_be_trainable:
                    conn.execute("""
                        UPDATE images SET is_trainable = 1 WHERE source_id = ?
                    """, (row['source_id'],))
                    stats['marked_trainable'] += 1
                    logger.debug(f"Marked {expected_filename} as trainable")
                else:
                    conn.execute("""
                        UPDATE images SET is_trainable = 0 WHERE source_id = ?
                    """, (row['source_id'],))
                    stats['marked_untrainable'] += 1
                    logger.debug(f"Marked {expected_filename} as untrainable")

        logger.info(f"Sync complete: {stats}")
        return stats

    def get_trainable(self,
                      source: Optional[ImageSource] = None,
                      label: Optional[ImageLabel] = None) -> List[ImageMetadata]:
        """
        Retrieve only trainable images.

        Args:
            source: Filter by image source (optional)
            label: Filter by image label (optional)

        Returns:
            List of ImageMetadata objects where is_trainable=True
        """
        query = "SELECT * FROM images WHERE is_trainable = 1"
        params = []

        if source is not None:
            query += " AND source = ?"
            params.append(source.value)

        if label is not None:
            query += " AND label = ?"
            params.append(label.value)

        query += " ORDER BY downloaded_at DESC"

        with self._get_connection() as conn:
            rows = conn.execute(query, params).fetchall()
            return [self._row_to_metadata(row) for row in rows]


    def count_trainable(self,
                        source: Optional[ImageSource] = None,
                        label: Optional[ImageLabel] = None) -> int:
        """
        Count trainable images.

        Args:
            source: Filter by image source (optional)
            label: Filter by image label (optional)

        Returns:
            Number of trainable images
        """
        query = "SELECT COUNT(*) FROM images WHERE is_trainable = 1"
        params = []

        if source is not None:
            query += " AND source = ?"
            params.append(source.value)

        if label is not None:
            query += " AND label = ?"
            params.append(label.value)

        with self._get_connection() as conn:
            return conn.execute(query, params).fetchone()[0]