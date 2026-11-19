from datetime import datetime

from common.config import UTC_TIMEZONE, JST_TIMEZONE


def get_dt_now_jst() -> datetime:
    """Get datetime now in JST"""
    return datetime.now(tz=JST_TIMEZONE)

def get_dt_now_utc() -> datetime:
    """Get datetime now in UTC"""
    return datetime.now(tz=UTC_TIMEZONE)
