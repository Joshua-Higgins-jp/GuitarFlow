from utils.dt_timestamps import get_dt_now_utc


def make_filename_pii_safe() -> str:
    """
    Generate a timestamp-based filename that contains no PII
    or metadata from the source image.
    """
    return get_dt_now_utc().strftime(format="%Y_%m_%d__%H_%M_%S")
