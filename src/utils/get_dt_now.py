from datetime import datetime
from zoneinfo import ZoneInfo


def get_dt_now_jst(tz: ZoneInfo = ZoneInfo("Asia/Tokyo")) -> datetime:
    """
    Returns the current datetime in JST.
    Returns the current datetime in UTC by default, but can handle user input timezones using ZoneInfo.

    To get UTC, use dt = get_dt_now(tz=ZoneInfo("UTC")) or simply call get_dt_now_utc()
    """
    return datetime.now(tz=tz)


def get_dt_now_utc() -> datetime:
    return get_dt_now_jst(tz=ZoneInfo("UTC"))
