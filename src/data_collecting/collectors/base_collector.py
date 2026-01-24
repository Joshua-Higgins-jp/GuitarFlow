from abc import ABC, abstractmethod
from pathlib import Path
from typing import List

from loguru import logger
from requests import get, Response, HTTPError

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
        return f"{self.service_name.value.lower()}_{metadata.image_id}.jpg"

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

    def download_single_image(
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
                url=metadata.image_url,
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
            logger.error(f"Failed to download {metadata.image_url}: {e}")
            return False
