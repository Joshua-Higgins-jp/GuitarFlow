from functools import cache
from os import getenv
from typing import Optional

from dotenv import find_dotenv, load_dotenv
from loguru import logger


@cache
def _load_dotenv_once() -> None:
    """
    Load the nearest .env file into the environment, once per process.

    The search starts at this file's directory and walks upward. Variables already present in the environment are
    never overridden, so values injected in production always take precedence.
    """
    dotenv_path: str = find_dotenv(raise_error_if_not_found=False)

    if dotenv_path:
        load_dotenv(dotenv_path=dotenv_path, override=False)
        logger.debug(f"Loaded environment variables from {dotenv_path}")


def load_env_var(key_name: str) -> str:
    """
    Read an environment variable, loading a .env file first if one is found.

    In prod, variables are expected to be injected directly into the environment. The .env file is only loaded
    when present (i.e. locally as DEV). If no .env file is found, the environment is read directly.

    Args:
        key_name: The environment variable name (e.g. API_KEY)

    Returns:
        The value associated with the given key, stripped of surrounding whitespace (guaranteed non-empty)

    Raises:
        KeyError: If the key is not found in the environment
        ValueError: If the key exists but is empty or contains only whitespace
    """
    _load_dotenv_once()

    value: Optional[str] = getenv(key=key_name)

    if value is None:
        raise KeyError(f"Environment variable `{key_name}` not found")

    value = value.strip()

    if not value:
        raise ValueError(f"Environment variable `{key_name}` exists, but is empty or contains only whitespace")

    return value
