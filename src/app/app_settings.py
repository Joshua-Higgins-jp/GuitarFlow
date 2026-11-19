from common.torch_devices import TorchDevices
from common.utils.read_dotenv import load_env_var


ENV: str = load_env_var(key_name="APP_ENV")

if ENV.upper() not in ["DEV", "PROD"]:
    raise RuntimeError(f"ENV '{ENV}' is not supported")

APP_DEBUG_MODE: bool = ENV.upper() == "DEV"
DATADOG_API_KEY: str = load_env_var(key_name="DATADOG_API_KEY")
DATADOG_SITE: str = load_env_var(key_name="DATADOG_SITE")


# For HuggingFace, Use CPU for free plan. GPU if you have money ;)
TORCH_DEVICE_OVERRIDE__APP: TorchDevices = TorchDevices.CPU
