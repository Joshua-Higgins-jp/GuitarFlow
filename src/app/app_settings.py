from os import getenv
from typing import Optional

from dotenv import load_dotenv, find_dotenv


load_dotenv(dotenv_path=find_dotenv(raise_error_if_not_found=True))

def load_env_var(key_name: str) -> str:
    """
    reads env variable values.
    make sure you run load_dotenv before this, otherwise your env
    won't even exist to read from.

    Args:
        key_name: the key name you want to assign

    Returns:
        env variable value

    Raises:
        RuntimeError: if env variable not found on the specified key.
    """
    env_var: Optional[str] = getenv(key=key_name)
    if not env_var:
        raise RuntimeError(f"Couldn't find env variable {key_name}. Unable to start")
    return env_var

ENV: str = load_env_var(key_name="ENV")
if ENV.upper() not in ["DEV", "PROD"]:
    raise RuntimeError(f"ENV {ENV} is not supported")

APP_DEBUG_MODE: bool = ENV.upper() == "DEV"

DATADOG_API_KEY: str = load_env_var(key_name="DATADOG_API_KEY")
