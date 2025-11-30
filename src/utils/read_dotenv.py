from os import getenv
from typing import Optional

from dotenv import load_dotenv, find_dotenv


def load_env_value_local(key_name: str) -> str:
    """
    Read a secret value from the local .env file
    """
    load_dotenv(dotenv_path=find_dotenv(raise_error_if_not_found=True))

    value: Optional[str] = getenv(key=key_name)
    if value is None:
        raise KeyError(f"Key `{key_name}` not found in environment")

    return value

#
# pixabay_api_key = load_env_value_local("PIXABAY_API_KEY")
# print(pixabay_api_key)
