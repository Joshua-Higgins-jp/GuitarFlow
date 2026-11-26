from dataclasses import dataclass
from datetime import datetime

from common.config import SESSION_ID_DT_STR_FORMAT, INGEST_RECONCILE_ID_FORMAT
from common.constants.job_names import JobNames


# TODO: consider having a session_id_base_str
#   then adding suffixes for specific places such as quarantine, ingestion, dir validation etc..
#   you should easily be able to define these here, but where used, just take in a string
#   such as session_id_str to avoid injecting as a dependency.
@dataclass(frozen=True)
class SessionID:
    session_start_time: datetime
    job_name: JobNames

    @property
    def session_id_str(self) -> str:
        """
        generates a session id for a given job type from the session start time.

        e.g. 20251201_002734_JST__updating_images
        """
        dt_part: str = self.session_start_time.strftime(format=SESSION_ID_DT_STR_FORMAT)
        return f"{dt_part}__{self.job_name.value}"

    @property
    def session_id_ingest_reconcile(self) -> str:
        """
        generates an ingest and reconcile session id from the start time

        e.g. 20251201_002734_JST
        """
        return self.session_start_time.strftime(format=INGEST_RECONCILE_ID_FORMAT)
