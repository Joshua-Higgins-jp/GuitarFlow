from os import environ
from json import dumps
from typing import Optional

from datadog_api_client import ApiClient, Configuration
from datadog_api_client.v2.api.logs_api import LogsApi
from datadog_api_client.v2.model.http_log import HTTPLog
from datadog_api_client.v2.model.http_log_item import HTTPLogItem
from loguru import logger
from pydantic import BaseModel


def _get_configuration() -> Configuration:
    """
    Build Datadog API configuration from environment variables.

    Expects DATADOG_API_KEY to be set in the environment.
    Optionally reads DATADOG_SITE (defaults to datadoghq.com).
    """
    api_key_raw: Optional[str]= environ.get("DATADOG_API_KEY")
    if not api_key_raw:
        raise EnvironmentError("DATADOG_API_KEY environment variable is not set.")

    config = Configuration()
    config.api_key["apiKeyAuth"] = api_key_raw

    site = environ.get("DATADOG_SITE", "datadoghq.com")
    config.server_variables["site"] = site

    return config


def _serialise_event(event: BaseModel) -> str:
    """
    Serialise a Pydantic model to a JSON string safe for Datadog.

    Handles non-serialisable types like datetime and UUID by
    converting them to strings via the default fallback.

    Args:
        event: Any Pydantic BaseModel instance.

    Returns:
        JSON string representation of the model.
    """
    return dumps(event.model_dump(), default=str)


def send_inference_event(event: BaseModel) -> None:
    """
    Send a single inference event to Datadog Log Management.

    Serialises the Pydantic model to structured JSON and submits
    it via the Datadog HTTP Logs API. The log will be queryable
    by any field in the Datadog UI, and can be used to build
    log-based metrics and dashboard panels.

    Args:
        event: A Pydantic BaseModel instance — expected to be
               InferenceEvent but accepts any model.

    Returns:
        None

    Example:
        event = InferenceEvent(
            session_id=uuid4(),
            model_version="v1.0",
            ...
        )
        send_inference_event(event)
    """
    try:
        config = _get_configuration()
        message = _serialise_event(event=event)
        ddtags: str = (
            f"env:{environ.get('ENVIRONMENT', 'prod')},"
            f"model_version:{getattr(event, 'model_version', 'unknown')}"
        )

        log_item = HTTPLogItem(
            ddsource="guitar-flow",
            ddtags=ddtags,
            hostname="hf-spaces",
            message=message,
            service="guitar-classifier",
        )

        with ApiClient(config) as api_client:
            api = LogsApi(api_client=api_client)
            api.submit_log(body=HTTPLog([log_item]))

        logger.info(
            f"Inference event sent to Datadog | class={getattr(event, 'predicted_class', '?')} "
            f"confidence={getattr(event, 'confidence_score', '?')}"
        )

    except EnvironmentError as env_err:
        logger.warning(f"Datadog not configured, skipping log | {env_err}")

    except Exception as e:
        logger.error(f"Failed to send inference event to Datadog | {e}")
