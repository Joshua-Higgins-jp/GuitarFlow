from dataclasses import dataclass
from datetime import datetime

from common.config import SESSION_ID_DT_STR_FORMAT


@dataclass(frozen=True)
class SessionID:
    session_start_time: datetime
    job_name: JobNames  # TODO: make JobNames StrEnum

    @property
    def session_id_str(self) -> str:
        """
        generates a session id for a given job type from the session start time.

        e.g. 20251201_002734_JST__updating_images
        """
        dt_part: str = self.session_start_time.strftime(format=SESSION_ID_DT_STR_FORMAT)
        return f"{dt_part}__{self.job_name.value}"
