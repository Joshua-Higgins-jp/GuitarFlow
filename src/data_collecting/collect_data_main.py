from pathlib import Path
from typing import List

from loguru import logger

from src.config import LOG_DIR
from src.data_collecting.collectors.collector_models import ImageMetadata, ImageLabel
from src.data_collecting.collectors.pixabay_collector import PixabayCollector
from src.data_collecting.data_dir_manager import DataDirectoryManager
from src.logging_manager import LoggerManager
from src.utils.read_dotenv import load_env_value_local


def main():
    logger_manager = LoggerManager(
        log_dir=LOG_DIR,
        session_name="downloading_images"
    )

    # Init The Data Directory Manager (basically just checks all required dirs exist before starting)
    dm = DataDirectoryManager()

    # Pixabay
    pixabay_api_key: str = load_env_value_local(key_name='PIXABAY_API_KEY')
    pixabay_collector = PixabayCollector(api_key=pixabay_api_key)
    pixabay_search_metadata: List[ImageMetadata] = pixabay_collector.search(
        query="electric guitar",
        label=ImageLabel.ELECTRIC,
        max_results=2
    )
    logger.info(f"Found {len(pixabay_search_metadata)} images")
    pixabay_downloaded_image_paths: List[Path] = pixabay_collector.download_images(metadata_list=pixabay_search_metadata)


    # Next service...



    # next service...


if __name__ == "__main__":
    main()
