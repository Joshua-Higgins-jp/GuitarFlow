import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Optional

from loguru import logger

from common.config import DB_TABLE_NAME
from data_management.db_spec import ColumnSpec, IndexSpec, EXPECTED_INDEXES, EXPECTED_COLUMNS, CREATE_STATEMENTS

"""
NOTE:
-- This code assumes there is only 1 table. If you ever expand to multiple tables, 
    pleas revisit this and remove the hard-coded table name
"""

class SchemaMismatchError(RuntimeError):
    """
    Raised when an existing database table does not match the expected schema.

    This usually means a schema change was made in code but the existing
    database file was never migrated. The database is left untouched.

    Attributes:
        problems: One human-readable line per detected difference.
    """
    def __init__(self, problems: list[str]) -> None:
        """
        Build the error from a list of detected schema differences.

        Args:
            problems: One human-readable line per difference found.
        """
        self.problems: list[str] = problems
        details: str = "\n  - ".join(problems)
        super().__init__(f"Table '{DB_TABLE_NAME}' does not match the expected schema:\n  - {details}")


class DatabaseManager:
    """
    Manages the SQLite database file backing the media processing queue.

    Call ensure_ready() once at application start. After that, use the
    connection() context manager to talk to the database.
    """
    def __init__(
            self,
            db_path: Path
    ) -> None:
        """
        Store the database location without touching the filesystem.

        Args:
            db_path: Location of the SQLite database file.
        """
        self.db_path: Path = db_path

    @contextmanager
    def connection(self) -> Iterator[sqlite3.Connection]:
        """
        Open a connection that commits on success and rolls back on error.

        Rows are returned as sqlite3.Row so columns can be read by name.
        The connection is always closed on exit.

        Yields:
            An open sqlite3.Connection.
        """
        conn: sqlite3.Connection = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()

        except Exception:
            conn.rollback()
            raise

        finally:
            conn.close()

    def ensure_ready(self) -> None:
        """
        Make sure the database file and queue table exist and are valid.

        Creates the file (and any missing parent directories) when absent.
        Creates the table when the file exists but has no table, which
        covers a first run that was interrupted before the schema was
        written. Always validates the table against the expected schema
        before returning.

        Raises:
            SchemaMismatchError: If the table exists but differs from the
                expected schema. No automatic migration is attempted.
        """
        file_existed: bool = self.db_path.exists(follow_symlinks=False)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        with self.connection() as conn:
            if not file_existed:
                logger.info(f"Database not found, creating new file at {self.db_path}")
                self._create_schema(conn)

            elif not self._table_exists(conn):
                logger.warning(
                    f"Database file {self.db_path} exists but has no "
                    f"'{DB_TABLE_NAME}' table, creating it"
                )
                self._create_schema(conn)

            else:
                logger.debug(f"Existing database found at {self.db_path}")

            self._validate_schema(conn)
            self._log_schema(conn)

        logger.info(f"Database ready, schema for '{DB_TABLE_NAME}' verified")

    @staticmethod
    def _table_exists(conn: sqlite3.Connection) -> bool:
        """
        Check whether the queue table is present in the database.

        Args:
            conn: An open connection.

        Returns:
            True if a table named TABLE_NAME exists.
        """
        row: Optional[sqlite3.Row] = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
            (DB_TABLE_NAME,)
        ).fetchone()
        return row is not None

    @staticmethod
    def _create_schema(conn: sqlite3.Connection) -> None:
        """
        Run every CREATE statement inside the current transaction.

        Statements are executed one at a time rather than with
        executescript, which would commit implicitly and break atomicity.

        Args:
            conn: An open connection.
        """
        for statement in CREATE_STATEMENTS:
            conn.execute(statement)

    @staticmethod
    def _read_columns(conn: sqlite3.Connection) -> dict[str, ColumnSpec]:
        """
        Read the actual column definitions of the queue table.

        Args:
            conn: An open connection.

        Returns:
            A mapping of column name to the ColumnSpec found in the database.
        """
        columns: dict[str, ColumnSpec] = {}
        for row in conn.execute(f"PRAGMA table_info({DB_TABLE_NAME})"):
            columns[row["name"]] = ColumnSpec(
                name=row["name"],
                sql_type=row["type"].upper(),
                not_null=bool(row["notnull"]),
                default=row["dflt_value"],
                is_pk=row["pk"] > 0,
            )
        return columns

    @staticmethod
    def _read_indexes(conn: sqlite3.Connection) -> set[IndexSpec]:
        """
        Read the actual indexes of the queue table, including UNIQUE
        constraint indexes that SQLite creates automatically.

        Args:
            conn: An open connection.

        Returns:
            The set of IndexSpec values found in the database.
        """
        indexes: set[IndexSpec] = set()
        for index_row in conn.execute(f"PRAGMA index_list({DB_TABLE_NAME})").fetchall():
            column_rows = conn.execute(
                f"PRAGMA index_info({index_row['name']})"
            ).fetchall()
            ordered_names: tuple[str, ...] = tuple(
                r["name"] for r in sorted(column_rows, key=lambda r: r["seqno"])
            )
            indexes.add(IndexSpec(ordered_names, bool(index_row["unique"])))
        return indexes

    def _validate_schema(self, conn: sqlite3.Connection) -> None:
        """
        Compare the actual table against the expected columns and indexes.

        Extra columns are treated as a mismatch. Extra indexes are allowed,
        but every expected index must be present.

        Args:
            conn: An open connection.

        Raises:
            SchemaMismatchError: If any difference is found. The error
                lists every difference, not just the first.
        """
        problems: list[str] = []

        expected: dict[str, ColumnSpec] = {c.name: c for c in EXPECTED_COLUMNS}
        actual: dict[str, ColumnSpec] = self._read_columns(conn)

        for name in sorted(expected.keys() - actual.keys()):
            problems.append(f"missing column: {name}")
        for name in sorted(actual.keys() - expected.keys()):
            problems.append(f"unexpected column: {name}")
        for name in sorted(expected.keys() & actual.keys()):
            if expected[name] != actual[name]:
                problems.append(
                    f"column {name}: expected [{expected[name].describe()}], "
                    f"found [{actual[name].describe()}]"
                )

        actual_indexes: set[IndexSpec] = self._read_indexes(conn)
        for index_spec in EXPECTED_INDEXES:
            if index_spec not in actual_indexes:
                problems.append(f"missing {index_spec.describe()}")

        if problems:
            logger.error(
                f"Schema validation failed for '{DB_TABLE_NAME}' "
                f"with {len(problems)} problem(s)"
            )
            raise SchemaMismatchError(problems)

    def _log_schema(self, conn: sqlite3.Connection) -> None:
        """
        Log the CREATE statements SQLite has stored for every table and index.

        This reads sqlite_master, so the output is the schema actually in
        the database file, not the schema the code intended to create.
        Tables are listed before indexes. Auto-generated indexes (such as
        those behind UNIQUE constraints) have no stored statement and are
        skipped, but the UNIQUE constraint itself is visible inside the
        owning table's CREATE statement.

        Args:
            conn: An open connection.
        """
        rows: list[sqlite3.Row] = conn.execute(
            "SELECT name, sql FROM sqlite_master "
            "WHERE sql IS NOT NULL AND name NOT LIKE 'sqlite_%' "
            "ORDER BY type DESC, name"
        ).fetchall()

        statements: str = "\n\n".join(row["sql"] for row in rows)
        logger.info(f"Database schema as stored in {self.db_path}:\n{statements}")

    @classmethod
    def init_database(cls, db_path: Path) -> 'DatabaseManager':
        """
        Create or verify the media queue database before any job runs.

        Returns:
            A MediaDatabase whose schema has been verified as valid.

        Raises:
            SystemExit: With code 1 if the existing schema does not match.
        """
        db = cls(db_path=db_path)
        try:
            db.ensure_ready()

        except SchemaMismatchError as exc:
            logger.error(f"Aborting startup, fix or migrate the database: {exc}")
            raise SystemExit(1) from exc

        return db
