from pathlib import Path

from common.constants.session_id import SessionID
from common.dirs_paths import _LOGS_DIR


def gen_log_filepath(session_id: SessionID) -> Path:
    """
    Loads the LOGS_DIR from _dirs to avoid triggering the logging of dir paths, which
    are intentionally lazy imported until after the logsink is made.

    e.g.
        log_20251201_002734_<tz>_<job_name>.log
    """
    return _LOGS_DIR / f"log__{session_id.session_id_job_name}.log"
