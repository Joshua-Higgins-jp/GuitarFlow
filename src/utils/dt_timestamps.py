from datetime import datetime
from zoneinfo import ZoneInfo


def get_dt_now_jst() -> datetime:
    return datetime.now(tz=ZoneInfo(key="Asia/Tokyo"))

def get_dt_now_utc() -> datetime:
    return datetime.now(tz=ZoneInfo(key="UTC"))
