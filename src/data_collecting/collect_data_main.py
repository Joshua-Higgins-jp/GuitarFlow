from pathlib import Path
from typing import List

from loguru import logger

from src.config import LOG_DIR
from src.data_collecting.collectors.collector_models_enums import ImageMetadata, ImageLabel
from src.data_collecting.collectors.pixabay_collector import PixabayCollector
from src.data_collecting.data_dir_manager import DataDirectoryManager, DATABASE_PATH
from src.data_collecting.image_metadata_db import ImageMetadataDB
from src.logging_manager import LoggerManager
from src.utils.read_dotenv import load_env_value_local


def data_collection_main():
    """
    The objective of this is to simply loop through 2 or 3 services and each time collect a new set of images

    Most of the free APIs have rate limits and hourly limits (like 50 images per hour or something) so multiple
    passes will be required for any given service to get all relevant images.
    """
    logger_manager = LoggerManager(
        log_dir=LOG_DIR,
        session_name="downloading_images"
    )

    # Init The Data Directory Manager (basically just checks all required dirs exist before starting)
    ddm = DataDirectoryManager()

    db = ImageMetadataDB(db_path=DATABASE_PATH)

    # Pixabay
    pixabay_api_key: str = load_env_value_local(key_name='PIXABAY_API_KEY')
    pixabay_collector = PixabayCollector(
        api_key=pixabay_api_key,
        per_page=50,
        min_width=640,
        min_height=480
    )
    pixabay_search_metadata: List[ImageMetadata] = pixabay_collector.search(
        query="electric guitar",
        label=ImageLabel.ELECTRIC,
        max_results=50
    )
    logger.info(f"Found {len(pixabay_search_metadata)} images")
    # Note that the database doesn't filter what to download - if an image already exists in the write dir,
    #   it gets skipped.
    pixabay_downloaded_image_metadata: List[ImageMetadata] = pixabay_collector.download_images(metadata_list=pixabay_search_metadata)

    # Take this list of successfully downloaded image metadata and write to database
    db.insert_batch(metadata_list=pixabay_downloaded_image_metadata)

    # Next service...



    # next service...


if __name__ == "__main__":
    data_collection_main()
