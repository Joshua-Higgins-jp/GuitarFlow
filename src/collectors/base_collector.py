from abc import ABC, abstractmethod
from hashlib import md5
from pathlib import Path
from time import sleep
from typing import List, Optional

from loguru import logger
from requests import get, Response

from src.collectors.collector_models import ImageMetadata
from src.config import RATE_LIMIT_SLEEP_BETWEEN_IMAGE_DOWNLOADS_SECS
from src.directory_manager import DATA_ROOT_DIR


class BaseCollector(ABC):
    """Abstract base for image collectors"""

    def __init__(
            self,
            api_key: str,
            service_name: str,
            data_dir: Path = DATA_ROOT_DIR
    ):
        self.api_key = api_key
        self.data_dir = data_dir
        self.service_name = service_name

    @abstractmethod
    def search(
            self,
            query: str,
            label: str,
            max_results: int = 100
    ) -> List[ImageMetadata]:
        """Search for images, return metadata list"""
        pass

    @abstractmethod
    def _build_search_url(self, query: str, page: int) -> str:
        """Build API request URL (service-specific)"""
        pass

    def download_images(
            self,
            metadata_list: List[ImageMetadata],
            delay_seconds: float = RATE_LIMIT_SLEEP_BETWEEN_IMAGE_DOWNLOADS_SECS
    ) -> List[Path]:
        """Download images, return saved paths"""

        # TODO: if an image already exists, skip the sleep.
        downloaded_paths: List[Path] = []

        for i, metadata in enumerate(metadata_list):
            logger.info(f"Downloading {i+1}/{len(metadata_list)}: {metadata.source_id}")

            path: Path = self._download_single_image(metadata=metadata)
            if path:
                downloaded_paths.append(path)

            # Rate limiting courtesy
            if i < len(metadata_list) - 1:
                sleep(delay_seconds)

        logger.info(f"Downloaded {len(downloaded_paths)}/{len(metadata_list)} images")
        return downloaded_paths

    def _download_single_image(
            self,
            metadata: ImageMetadata
    ) -> Optional[Path]:
        """Download one image, return path or None"""
        try:
            # Build filename: {service}_{source_id}.jpg
            filename: str = f"{self.service_name}_{metadata.source_id}.jpg"

            # Determine subdirectory
            label_dir: Path = self.data_dir / "raw" / self.service_name.lower() / metadata.label.value.lower()
            label_dir.mkdir(parents=True, exist_ok=True)
            filepath: Path = label_dir / filename

            # Skip if already exists
            if filepath.exists():
                logger.debug(f"Image already exists: {filepath}")
                return filepath

            # Download
            response: Response = get(
                url=metadata.url,
                timeout=30,
                stream=True
            )
            response.raise_for_status()

            # Save (avoids memory writes)
            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            logger.info(f"Downloaded: {filepath}")
            return filepath

        except Exception as e:
            logger.error(f"Failed to download {metadata.url}: {e}")
            return None

    @staticmethod
    def _compute_md5(file_path: Path) -> str:
        """Compute MD5 hash for deduplication"""
        hash_md5 = md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
