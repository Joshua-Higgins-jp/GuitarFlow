from dataclasses import dataclass
from datetime import datetime

from common.config import SESSION_ID_DT_STR_FORMAT
from common.constants.job_names import JobNames


@dataclass(frozen=True)
class SessionID:
    session_start_time: datetime
    job_name: JobNames

    @property
    def session_id_base(self) -> str:
        return self.session_start_time.strftime(format=SESSION_ID_DT_STR_FORMAT)

    @property
    def session_id_job_name(self) -> str:
        return f"{self.session_id_base}__{self.job_name.value}"

    @property
    def session_id_quarantine(self) -> str:
        return f"{self.session_id_base}__quarantine.csv"
