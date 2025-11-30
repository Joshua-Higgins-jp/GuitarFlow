from datetime import datetime
from zoneinfo import ZoneInfo


def get_dt_now(tz: ZoneInfo = ZoneInfo("UTC")) -> datetime:
    """
    Returns the current datetime in UTC by default, but can handle user input timezones using ZoneInfo.

    To get JST (Japan standard time), use dt = get_dt_now(tz=ZoneInfo("Asia/Tokyo"))
    """
    return datetime.now(tz=tz)


# print(get_dt_now())
# print(get_dt_now(tz=ZoneInfo("Asia/Tokyo")))
