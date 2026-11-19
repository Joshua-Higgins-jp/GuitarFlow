from datetime import datetime

from common.config import PII_SAFE_FORMAT


def gen_pii_safe_filename(dt: datetime) -> str:
    """
    Generate a datetime timestamp-based filename that contains no PII
    or metadata from the source image.
    """
    return dt.strftime(format=PII_SAFE_FORMAT)
