from os import getenv
from typing import Optional

from dotenv import load_dotenv, find_dotenv


def load_env_value_local(key_name: str) -> str:
    """
    Read a secret value from the local .env file.

    Args:
        key_name: The environment variable name (e.g., 'API_KEY')

    Returns:
        The value associated with the given key (guaranteed non-None, non-empty)

    Raises:
        KeyError: If the key is not found in the environment
        ValueError: If the key exists but is empty
    """
    load_dotenv(dotenv_path=find_dotenv(raise_error_if_not_found=True))

    value: Optional[str] = getenv(key=key_name)

    if value is None:
        raise KeyError(f"Environment variable `{key_name}` not found")

    if not value:  # Catches empty strings
        raise ValueError(f"Environment variable `{key_name}` exists, but is assigned an empty string")

    return value
