from typing import Optional
from zoneinfo import ZoneInfo

from common.constants.class_labels import ClassLabels
from common.constants.torch_devices import TorchDevices

JST_TIMEZONE = ZoneInfo(key="Asia/Tokyo")
UTC_TIMEZONE = ZoneInfo(key="UTC")

PII_SAFE_DT_FORMAT: str = "%Y_%m_%d__%H_%M_%S"
SESSION_ID_DT_STR_FORMAT = "%Y%m%d_%H%M%S_%Z"


DIRS_PATHS_LOG_NAME_WIDTH: int = 23
DB_NAME: str = "image_dataset.sqlite3"

CLASS_NAMES_LIST: list[str] = ClassLabels.as_list()
CLASS_TO_IDX: dict[str, int] = {name: idx for idx, name in enumerate(CLASS_NAMES_LIST)}

IMAGENET_MEAN: list[float] = [0.485, 0.456, 0.406]
IMAGENET_STD: list[float]  = [0.229, 0.224, 0.225]
IMG_SIZE: tuple[int, int]  = (224, 224)


TORCH_DEVICE_OVERRIDE__LOCAL: Optional[TorchDevices] = TorchDevices.MPS  # for mac
