from pathlib import Path

from loguru import logger


def add_log_sink(log_filepath: Path) -> None:
    """
    Saves a .log file to logs_dir using filename session_id

    Returns the Path for this session's logger file.
    """
    logger.add(
        sink=str(log_filepath),
        backtrace=True,
        diagnose=True,
        level="DEBUG",
    )
