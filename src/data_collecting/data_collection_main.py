from typing import List, Tuple

from data_collecting.collectors.collector_models_enums import ImageLabel, ImageMetadata, ImageSource
from data_collecting.collectors.image_download_manager import ImageDownloadManager
from data_collecting.collectors.pixabay.pixabay_collector import PixabayCollector
from src.config import LOG_DIR
from src.data_collecting.collector_database import ImageMetadataDBManager
from src.data_collecting.data_dir_manager import DataDirectoryManager, DATABASE_PATH
from src.logging_manager import LoggerManager
from utils.read_dotenv import load_env_value_local


def data_collection_main():
    """
    The objective of this is to simply loop through 2 or 3 services and each time collect a new set of images

    Most of the free APIs have rate limits and hourly limits (like 50 images per hour or something) so multiple
    passes will be required for any given service to get all relevant images.
    """
    LoggerManager(
        log_dir=LOG_DIR,
        session_name="downloading_images"
    )
    data_dir_manager = DataDirectoryManager()
    db_manager = ImageMetadataDBManager(db_path=DATABASE_PATH)
    image_download_manager = ImageDownloadManager(db_manager=db_manager)

    # API Keys, Config, and Inits
    pixabay_api_key: str = load_env_value_local(key_name='PIXABAY_API_KEY')

    # Queries and Labels
    pixabay_queries_and_labels: List[Tuple[str, ImageLabel]] = [
        ("electric guitar", ImageLabel.ELECTRIC),
        ("acoustic guitar", ImageLabel.ACOUSTIC)
    ]

    # Collector Inits
    pixabay_collector = PixabayCollector(
        api_key=pixabay_api_key,
        results_per_page=200,
        min_width=640,
        min_height=480
    )

    pixabay_collector.query_and_update_database(
        database_manager=db_manager,
        queries_and_labels=pixabay_queries_and_labels
    )

    # Start Downloading Files
    list_of_pending_downloads: List[ImageMetadata] = db_manager.get_pending_downloads(
        limit=50,  # conservative starting point
        source=ImageSource.PIXABAY,
        label=ImageLabel.ELECTRIC,
    )

    image_download_manager.download_batch(
        metadata_list=list_of_pending_downloads,
        skip_already_downloaded=True,
    )


if __name__ == "__main__":
    data_collection_main()
