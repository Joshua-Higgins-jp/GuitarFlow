from os import getenv
from typing import Optional

from dotenv import load_dotenv, find_dotenv


def load_env_var(key_name: str) -> str:
    """
    Read an environment variable, optionally loading from a .env file.

    In production environments, variables are expected to be injected directly
    into the environment. The .env file is only loaded when present (i.e. local
    development). If no .env file is found, the function falls back to reading
    from the environment directly.

    Args:
        key_name: The environment variable name (e.g. API_KEY)

    Returns:
        The value associated with the given key (guaranteed non-None, non-empty)

    Raises:
        KeyError: If the key is not found in the environment
        ValueError: If the key exists but is empty or contains only whitespace
    """
    dotenv_path: str = find_dotenv(
        filename=".env",
        raise_error_if_not_found=False
    )

    if dotenv_path:
        load_dotenv(dotenv_path=dotenv_path, verbose=True)

    value: Optional[str] = getenv(key=key_name)

    if value is None:
        raise KeyError(f"Environment variable `{key_name}` not found")

    value = value.strip()

    if not value:
        raise ValueError(f"Environment variable `{key_name}` exists, but is empty or contains only whitespace")

    return value
