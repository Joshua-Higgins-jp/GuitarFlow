from datetime import datetime, timezone

def get_dt_now(tz: timezone = timezone.utc) -> datetime:
    return datetime.now(tz=tz)
