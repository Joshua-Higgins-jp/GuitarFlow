from pathlib import Path
from typing import List

from src.collectors.collector_models import ImageMetadata, ImageLabel
from src.collectors.pixabay import PixabayCollector
from src.directory_manager import DirectoryManager
from src.utils.read_dotenv import load_env_value_local
from loguru import logger


dm = DirectoryManager()
pixabay_collector = PixabayCollector(api_key=load_env_value_local(key_name='PIXABAY_API_KEY'))

# Test search - small sample first
search_metadata: List[ImageMetadata] = pixabay_collector.search(
    query="electric guitar",
    label=ImageLabel.ELECTRIC,
    max_results=50
)

logger.info(f"\nFound {len(search_metadata)} images")

pixabay_download_paths: List[Path] = pixabay_collector.download_images(search_metadata)


# # Test download
# if search_metadata:
#     paths = pixabay_collector.download_images(search_metadata)
#     print(f"Downloaded {len(paths)} images")
#     print(f"Saved to: {paths[0].parent}")
