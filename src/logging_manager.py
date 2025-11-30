import sys
from pathlib import Path
from zoneinfo import ZoneInfo

from loguru import logger

from src.utils.get_dt_now import get_dt_now


class LoggerManager:
    """
    Manages session-based logging to plain text files.

    Each session gets its own timestamped .log file.
    Keep it simple - convert to HTML later if needed.

    Attributes:
        class_log_pfx: Logging prefix for all class operations
        log_dir: Directory where log files are stored
        session_name: Name of the current session
        log_path: Path to the current session's log file
    """
    def __init__(
            self,
            log_dir: Path,
            session_name: str,
            console_level: str = "DEBUG",
            file_level: str = "TRACE"
    ) -> None:
        """
        Initialize the LoggerManager for a new session.

        Args:
            log_dir: Directory to save log files
            session_name: Name for this session (used in filename)
            console_level: Logging level for console output (DEBUG/INFO/WARNING/ERROR)
            file_level: Logging level for file output
        """
        self.class_log_pfx: str = "LOGGER MANAGER"
        self.log_dir: Path = log_dir
        self.session_name: str = session_name

        # Create log directory
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Generate timestamp for this session
        timestamp: str = get_dt_now(tz=ZoneInfo('Asia/Tokyo')).strftime("%Y%m%d_%H%M%S")

        # Create filename (includes JST suffix)
        self.log_path: Path = self.log_dir / f"log_{timestamp}_JST_{session_name}.log"

        # Log format
        self.log_format: str = (
            "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        )

        # Store handler IDs for cleanup
        self.handler_ids: list[int] = []

        # Setup logging
        self._setup_logging(console_level, file_level)

        logger.info(f"{self.class_log_pfx}: Session '{session_name}' logging initialized")
        logger.info(f"{self.class_log_pfx}: Log file: {self.log_path}")

    def _setup_logging(self, console_level: str, file_level: str) -> None:
        """
        Configure loguru sinks for console and file output.

        Args:
            console_level: Logging level for console
            file_level: Logging level for files
        """
        # Remove default handler
        logger.remove()

        # Add console handler with colors
        console_id = logger.add(
            sys.stderr,
            format=self.log_format,
            level=console_level,
            colorize=True
        )
        self.handler_ids.append(console_id)

        # Add file handler (plain text, no ANSI codes)
        file_id = logger.add(
            self.log_path,
            format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} | {message}",
            level=file_level,
            colorize=False,
            enqueue=True
        )
        self.handler_ids.append(file_id)

    def close(self) -> None:
        """
        Close all log handlers.

        Call this at the end of your session to properly close log files.
        """
        logger.info(f"{self.class_log_pfx}: Closing session '{self.session_name}' logging")

        # Remove all handlers
        for handler_id in self.handler_ids:
            logger.remove(handler_id)

        self.handler_ids.clear()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - automatically closes logging."""
        self.close()