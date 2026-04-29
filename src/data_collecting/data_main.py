from data_collecting.ingest_images import ImageIngester
from data_collecting.metadata_models_enums import ImageSource
from config.globals import ClassLabels
from src.config import LOG_DIR
from src.data_collecting.collector_database import ImageMetadataDBManager
from src.data_collecting.data_dir_manager import DataDirectoryManager, DATABASE_PATH
from monitoring.logging_manager import LoggerManager


def data_ingestion_main():

    logger_manager = LoggerManager(
        log_dir=LOG_DIR,
        session_name="downloading_images"
    )
    data_dir_manager = DataDirectoryManager()
    image_metadata_manager = ImageMetadataDBManager(db_path=DATABASE_PATH)
    image_ingester_manager = ImageIngester(db_manager=image_metadata_manager)
    unsplash_acoustic_ingest = image_ingester_manager.ingest_directory(
        directory=data_dir_manager.get_raw_dir(
            image_source=ImageSource.UNSPLASH,
            label=ClassLabels.ACOUSTIC
        ),
        source=ImageSource.UNSPLASH,
        label=ClassLabels.ACOUSTIC
    )
    unsplash_electric_ingest = image_ingester_manager.ingest_directory(
        directory=data_dir_manager.get_raw_dir(
            image_source=ImageSource.UNSPLASH,
            label=ClassLabels.ELECTRIC
        ),
        source=ImageSource.UNSPLASH,
        label=ClassLabels.ELECTRIC
    )


if __name__ == "__main__":
    data_ingestion_main()
