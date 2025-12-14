from abc import ABC, abstractmethod
from hashlib import md5
from pathlib import Path
from time import sleep
from typing import List, Union

from loguru import logger
from requests import get, Response, HTTPError

from src.config import RATE_LIMIT_SLEEP_BETWEEN_IMAGE_DOWNLOADS_SECS
from src.data_collecting.collectors.collector_models_enums import ImageMetadata, ImageSource
from src.data_collecting.data_dir_manager import DATA_ROOT_DIR


class BaseCollector(ABC):
    """Abstract base for image collectors"""

    def __init__(
            self,
            api_key: str,
            service_name: ImageSource,
            data_dir: Path = DATA_ROOT_DIR
    ):
        if not isinstance(api_key, str):
            raise TypeError("API key must be a string")

        if not isinstance(service_name, ImageSource):
            raise TypeError("Service name must be of ImageSource object")

        if not isinstance(data_dir, Path):
            raise TypeError("data_dir must be a Path object")

        self.api_key: str = api_key
        self.data_dir: Path = data_dir
        self.service_name: ImageSource = service_name

    @abstractmethod
    def search(
            self,
            query: str,
            label: str,
            max_results: int
    ) -> List[ImageMetadata]:
        """Search for images, return metadata list"""
        pass

    @abstractmethod
    def _build_search_url(
            self,
            query: str,
            page: int
    ) -> str:
        """Build API request URL (service-specific)"""
        pass

    def _generate_filename(self, metadata: ImageMetadata) -> str:
        """
        Generate filename for an image based on its metadata.

        Args:
            metadata: ImageMetadata object containing source and ID info

        Returns:
            Filename string in format: {service}_{source_id}.jpg
        """
        return f"{self.service_name.value.lower()}_{metadata.source_id}.jpg"

    def _get_filepath(self, metadata: ImageMetadata) -> Path:
        """
        Get the full filepath where an image should be saved.

        Args:
            metadata: ImageMetadata object

        Returns:
            Full Path object: data_dir/raw/{service}/{label}/{filename}
        """
        filename: str = self._generate_filename(metadata=metadata)
        label_dir: Path = self.data_dir / "raw" / self.service_name.value.lower() / metadata.label.value.lower()
        return label_dir / filename

    def download_images(
            self,
            metadata_list: List[ImageMetadata],
            delay_seconds: Union[int, float] = RATE_LIMIT_SLEEP_BETWEEN_IMAGE_DOWNLOADS_SECS
    ) -> List[ImageMetadata]:
        """
        Download images from metadata list.

        Args:
            metadata_list: List of ImageMetadata objects to download
            delay_seconds: Time to wait between downloads for rate limiting

        Returns:
            List of ImageMetadata for successfully downloaded images
        """
        downloaded_metadata: List[ImageMetadata] = []

        for i, metadata in enumerate(metadata_list):
            logger.info(f"Processing {i + 1}/{len(metadata_list)}: Image ID {metadata.source_id}")

            # Get filepath and check existence
            filepath: Path = self._get_filepath(metadata=metadata)
            filepath.parent.mkdir(parents=True, exist_ok=True)

            # Skip if already exists
            if filepath.exists():
                logger.warning(f"Image already exists, skipping: {filepath}")
                downloaded_metadata.append(metadata)
                continue  # Skip sleep since we didn't hit the API

            # Download the image
            success: bool = self._download_single_image(metadata=metadata)
            if success:
                downloaded_metadata.append(metadata)

                # Rate limiting courtesy - only after actual download
                if i < len(metadata_list) - 1:
                    sleep(delay_seconds)

        logger.info(f"Downloaded {len(downloaded_metadata)}/{len(metadata_list)} images")
        return downloaded_metadata

    def _download_single_image(
            self,
            metadata: ImageMetadata
    ) -> bool:
        """
        Download a single image from metadata.

        Args:
            metadata: ImageMetadata containing URL and save location info

        Returns:
            True if download succeeded, False otherwise
        """
        try:
            # Get filepath using centralized method
            filepath: Path = self._get_filepath(metadata=metadata)

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
            return True

        except HTTPError as http_err:
            logger.error(f"HTTP error occurred: {http_err}")
            return False

        except Exception as e:
            logger.error(f"Failed to download {metadata.url}: {e}")
            return False

    @staticmethod
    def _compute_md5(file_path: Path) -> str:
        """Compute MD5 hash for deduplication"""
        hash_md5 = md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
