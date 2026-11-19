from zoneinfo import ZoneInfo


JST_TIMEZONE = ZoneInfo(key="Asia/Tokyo")
UTC_TIMEZONE = ZoneInfo(key="UTC")

PII_SAFE_FORMAT: str = "%Y_%m_%d__%H_%M_%S"

DIRS_PATHS_LOG_NAME_WIDTH: int = 23
DB_NAME: str = "image_dataset.sqlite3"