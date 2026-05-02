from datadog_api_client import ApiClient, Configuration
from datadog_api_client.v2.api.logs_api import LogsApi
from datadog_api_client.v2.model.http_log import HTTPLog
from datadog_api_client.v2.model.http_log_item import HTTPLogItem
from loguru import logger

from schemas.inference_event import InferenceEvent
from app_settings import ENV, DATADOG_API_KEY, DATADOG_SITE


def _get_configuration() -> Configuration:
    """
    Build Datadog API configuration from application settings.

    DATADOG_API_KEY and DATADOG_SITE are validated at startup in
    app_settings and guaranteed to be present by the time this is called.

    Returns:
        A Configuration instance ready for use with ApiClient.
    """
    config = Configuration()
    config.api_key["apiKeyAuth"] = DATADOG_API_KEY
    config.server_variables["site"] = DATADOG_SITE
    return config


def send_inference_event(event: InferenceEvent) -> None:
    """
    NOTE: THIS IS SYNCHRONOUS, NOT ASYNC.

    Send a single inference event to Datadog Log Management.

    Serialises the InferenceEvent to structured JSON and submits it
    via the Datadog HTTP Logs API. The log is queryable by any field
    in the Datadog Log Explorer and can be used to build log-based
    metrics and dashboard panels.

    A new ApiClient is instantiated per call. This is acceptable for
    the expected inference volume of Guitar Flow but should be replaced
    with a persistent client if call frequency increases significantly.

    Args:
        event: The InferenceEvent produced by one classification inference.

    Returns:
        None

    Example:
        event = InferenceEvent(
            session_id=uuid4(),
            model_version="v1.0",
            predicted_class=ClassLabels.ELECTRIC,
            ...
        )
        send_inference_event(event)
    """
    try:
        config: Configuration = _get_configuration()
        message: str = event.model_dump_json()
        ddtags: str = (
            f"env:{ENV.lower()},"
            f"model_version:{event.model_version}"
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
            f"Inference event sent to Datadog | "
            f"class={event.predicted_class} "
            f"confidence={event.confidence_score} "
            f"latency={event.inference_latency_ms}ms"
        )

    except Exception as e:
        logger.error(f"Failed to send inference event to Datadog | cause={e}")
