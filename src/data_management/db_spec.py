"""
Schema of the image records table: one row per image that has passed the dataset checks at least once.

The columns match ImageRecordModel field for field. Its docstring explains why each field exists and why some were
dropped. The table enforces only what must always be true (NOT NULL, positive sizes, a 64 character hash, a valid
EXIF orientation, one ACTIVE row per path). It does not restrict labels, sources, modes or formats to fixed lists, so
adding one never needs a migration, and an unexpected value never stops a scan.
"""
from dataclasses import dataclass
from typing import Optional

from common.config import DB_TABLE_NAME
from common.constants.status_labels import StatusLabels


@dataclass(frozen=True)
class ColumnSpec:
    """
    Expected definition of one table column, as reported by PRAGMA table_info.

    CHECK constraints are not reported by PRAGMA table_info, so they are not part of this spec and are not validated.

    Attributes:
        name: Column name.
        sql_type: Declared SQL type in upper case, for example "TEXT".
        not_null: True if the column carries a NOT NULL constraint.
        default: Default value exactly as SQLite reports it (string literals include their single quotes), or None
            for no default.
        is_pk: True if the column is part of the primary key.
    """
    name: str
    sql_type: str
    not_null: bool
    default: Optional[str]
    is_pk: bool

    def describe(self) -> str:
        """
        Build a short human-readable summary of this column definition.

        Returns:
            A single-line description used in mismatch error messages.
        """
        return f"{self.sql_type} not_null={self.not_null} default={self.default} pk={self.is_pk}"


@dataclass(frozen=True)
class IndexSpec:
    """
    Expected definition of one index, identified by columns, uniqueness and whether it is partial.

    The index name is intentionally ignored. A UNIQUE column constraint creates an auto-named index, and only its
    behaviour matters here. For a partial index only the fact that it has a WHERE clause is checked, not the clause.

    Attributes:
        columns: Ordered column names covered by the index.
        unique: True if the index enforces uniqueness.
        partial: True if the index only covers rows matching a WHERE clause.
    """
    columns: tuple[str, ...]
    unique: bool
    partial: bool = False

    def describe(self) -> str:
        """
        Build a short human-readable summary of this index definition.

        Returns:
            A single-line description used in mismatch error messages.
        """
        kind: str = "unique index" if self.unique else "index"
        prefix: str = "partial " if self.partial else ""
        return f"{prefix}{kind} on ({', '.join(self.columns)})"


EXPECTED_COLUMNS: tuple[ColumnSpec, ...] = (
    # set once, on insert
    ColumnSpec("pixel_hash", "TEXT", True, None, True),
    ColumnSpec("width", "INTEGER", True, None, False),
    ColumnSpec("height", "INTEGER", True, None, False),
    ColumnSpec("ingested_at", "TEXT", True, None, False),
    # refreshed every scan
    ColumnSpec("label", "TEXT", True, None, False),
    ColumnSpec("source", "TEXT", True, None, False),
    ColumnSpec("filename", "TEXT", True, None, False),
    ColumnSpec("mode", "TEXT", True, None, False),
    ColumnSpec("image_format", "TEXT", True, None, False),
    ColumnSpec("exif_orientation", "INTEGER", True, None, False),
    ColumnSpec("filesize_bytes", "INTEGER", True, None, False),
    ColumnSpec("checks_version", "TEXT", True, None, False),
    ColumnSpec("status", "TEXT", True, None, False),
    ColumnSpec("last_seen_at", "TEXT", True, None, False),
)

COLUMN_NAMES: tuple[str, ...] = tuple(column.name for column in EXPECTED_COLUMNS)

# Every other column is set once on insert and never changed by an update
REFRESHED_COLUMNS: tuple[str, ...] = (
    "label",
    "source",
    "filename",
    "mode",
    "image_format",
    "exif_orientation",
    "filesize_bytes",
    "checks_version",
    "status",
    "last_seen_at",
)

EXPECTED_INDEXES: tuple[IndexSpec, ...] = (
    IndexSpec(("label", "source", "filename"), unique=True, partial=True),
)

CREATE_STATEMENTS: tuple[str, ...] = (
    f"""
    CREATE TABLE {DB_TABLE_NAME} (
        pixel_hash          TEXT    NOT NULL PRIMARY KEY CHECK (length(pixel_hash) = 64),  -- SHA-256 of decoded pixels
        width               INTEGER NOT NULL CHECK (width > 0),  -- after EXIF orientation is applied
        height              INTEGER NOT NULL CHECK (height > 0),  -- after EXIF orientation is applied
        ingested_at         TEXT    NOT NULL,  -- ISO 8601 datetime, JST. Set once on insert
        label               TEXT    NOT NULL,  -- folder name, images_raw/<label>/<source>/<filename>
        source              TEXT    NOT NULL,  -- folder name
        filename            TEXT    NOT NULL,  -- current filename, including the extension
        mode                TEXT    NOT NULL,  -- PIL mode as stored in the file, before conversion to RGB
        image_format        TEXT    NOT NULL,  -- format detected from the content, MPO reported as JPEG
        exif_orientation    INTEGER NOT NULL CHECK (exif_orientation BETWEEN 1 AND 8),
        filesize_bytes      INTEGER NOT NULL CHECK (filesize_bytes > 0),
        checks_version      TEXT    NOT NULL,  -- ImageCheckConfig.version of the scan that last checked the image
        status              TEXT    NOT NULL,  -- ACTIVE or MISSING
        last_seen_at        TEXT    NOT NULL   -- ISO 8601 datetime, JST. Last scan that found the image and passed it
    )
    """,
    # Two ACTIVE rows can never point at the same file, but a MISSING row's old path can be reused by a new image
    f"CREATE UNIQUE INDEX idx_{DB_TABLE_NAME}_active_path ON {DB_TABLE_NAME}(label, source, filename) "
    f"WHERE status = '{StatusLabels.ACTIVE.value}'",
)
