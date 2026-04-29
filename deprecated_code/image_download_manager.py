from pathlib import Path
from random import uniform
from time import sleep
from typing import List, Optional

from PIL import Image
from loguru import logger
from pydantic import BaseModel, Field
from requests import get, Response, HTTPError

from src.data_collecting.collector_database import ImageMetadataDBManager
from src.data_collecting.collectors.collector_models_enums import ImageMetadata, ImageSource, ImageLabel
from src.data_collecting.data_dir_manager import DATA_ROOT_DIR
from utils.get_filesize_bytes import get_filesize_bytes


class DownloadResult(BaseModel):
    success: int = Field(ge=0)
    failed: int = Field(ge=0)
    skipped: int = Field(ge=0)


class ImageDownloadManager:
    """
    Manages batch downloading of images from metadata and updates database status.

    Handles rate limiting, retries, and atomic database updates.
    """
    def __init__(
            self,
            db_manager: ImageMetadataDBManager,
            data_dir: Path = DATA_ROOT_DIR,
            timeout: int = 30,
            min_sleep: float = 15.0,
            max_sleep: float = 30.0,
            max_retries: int = 3
    ) -> None:
        """
        Initialise the image download manager.

        Args:
            db_manager: Database manager for updating download status
            data_dir: Root directory for storing downloaded images
            timeout: Request timeout in seconds (default: 30)
            min_sleep: Minimum sleep between downloads in seconds (default: 15.0)
            max_sleep: Maximum sleep between downloads in seconds (default: 30.0)
            max_retries: Maximum retry attempts for failed downloads (default: 3)
        """
        if not isinstance(db_manager, ImageMetadataDBManager):
            raise TypeError("db_manager must be an ImageMetadataDBManager object")

        if not isinstance(data_dir, Path):
            raise TypeError("data_dir must be a Path object")

        if not isinstance(timeout, int) or timeout <= 0:
            raise ValueError("timeout must be a positive integer")

        if min_sleep < 0 or max_sleep < min_sleep:
            raise ValueError("Invalid sleep range: min_sleep must be >= 0 and <= max_sleep")

        self.db_manager: ImageMetadataDBManager = db_manager
        self.data_dir: Path = data_dir
        self.timeout: int = timeout
        self.min_sleep: float = min_sleep
        self.max_sleep: float = max_sleep
        self.max_retries: int = max_retries

    def _get_filepath(self, metadata: ImageMetadata) -> Path:
        """
        Generate the full filepath where an image should be saved.

        Args:
            metadata: ImageMetadata object

        Returns:
            Full Path: data_dir/raw/{source}/{label}/{source}_{image_id}.jpg
        """
        filename: str = f"{metadata.source.value.lower()}_{metadata.image_id}.jpg"
        label_dir: Path = (
                self.data_dir / "raw" /
                metadata.source.value.lower() /
                metadata.label.value.lower()
        )

        # Ensure directory exists
        label_dir.mkdir(parents=True, exist_ok=True)

        return label_dir / filename

    def _download_single_image(
            self,
            image_metadata: ImageMetadata,
            destination_filepath: Path
    ) -> bool:
        """
        Download a single image from URL to filepath.

        Args:
            image_metadata: information about a specific image
            destination_filepath: Destination path for the image

        Returns:
            True if download succeeded, False otherwise
        """
        try:
            response: Response = get(
                url=image_metadata.image_url,
                timeout=self.timeout,
                stream=True
            )
            response.raise_for_status()

            if response.status_code == 429:
                logger.warning("Rate limit hit! Consider increasing sleep time or reducing batch size")
                # retry_after = response.headers.get('Retry-After', '300')
                # logger.warning(f"Suggested retry after: {retry_after} seconds")
                return False

            # Save image in chunks to avoid memory issues
            with open(destination_filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)

            # Verify it's a valid image
            try:
                with Image.open(destination_filepath) as img:
                    width, height = img.size
                    logger.debug(f"Verified image: {width}x{height}")

            except Exception as img_err:
                logger.warning(f"Downloaded file may be corrupted: {img_err}")
                destination_filepath.unlink(missing_ok=True)  # Delete corrupted file
                return False

            logger.info(f"Downloaded: {image_metadata.image_id} → {destination_filepath.name}")
            return True

        except HTTPError as http_err:
            logger.error(f"HTTP error for {image_metadata.image_id}: {http_err}")
            return False

        except Exception as e:
            logger.error(f"Failed to download {image_metadata.image_id}: {e}")
            return False

    def _random_sleep(self) -> None:
        """Sleep for a random duration between min_sleep and max_sleep."""
        sleep_duration: int = int(uniform(a=self.min_sleep, b=self.max_sleep))
        logger.debug(f"Sleeping for {sleep_duration} seconds...")
        sleep(sleep_duration)

    def download_batch(
            self,
            metadata_list: List[ImageMetadata],
            skip_already_downloaded: bool = True
    ) -> DownloadResult:
        """
        Download a batch of images and update database status.

        Args:
            metadata_list: List of ImageMetadata objects to download
            skip_already_downloaded: If True, skip images already marked as downloaded

        Returns:
            DownloadResult object storing success, failed, and skipped info for the run.
        """
        success_count: int = 0
        failed_count: int = 0
        skipped_count: int = 0

        total: int = len(metadata_list)
        logger.info(f"Starting batch download: {total} images")

        for idx, metadata in enumerate(metadata_list, start=1):
            logger.info(f"Processing {idx}/{total}: {metadata.image_id}")

            # Skip if already downloaded
            if skip_already_downloaded and metadata.is_downloaded:
                logger.debug(f"Skipping {metadata.image_id} - already downloaded")
                skipped_count += 1
                continue

            filepath: Path = self._get_filepath(metadata=metadata)

            # Skip if file already exists on disk
            if filepath.exists():
                logger.debug(f"File exists on disk: {filepath.name}, updating database...")
                file_size: int = get_filesize_bytes(filepath=filepath)

                # Get image dimensions
                try:
                    with Image.open(filepath) as img:
                        width, height = img.size
                except Exception as e:
                    logger.error(f"Failed to open image {metadata.image_id}: {e}")
                    width, height = None, None

                self.db_manager.update_download_status(
                    image_id=metadata.image_id,
                    width=width,
                    height=height,
                    filesize=file_size
                )
                skipped_count += 1
                continue

            # Attempt download with retries
            success: bool = False
            for attempt in range(1, self.max_retries + 1):
                if attempt > 1:
                    logger.info(f"Retry attempt {attempt}/{self.max_retries} for {metadata.image_id}")

                success: bool = self._download_single_image(
                    image_metadata=metadata,
                    destination_filepath=filepath
                )

                if success:
                    break

                if attempt < self.max_retries:
                    sleep(5)  # Short sleep between retries

            if success:
                # Get actual file info
                file_size: int = get_filesize_bytes(filepath=filepath)

                try:
                    with Image.open(filepath) as img:
                        width, height = img.size
                except Exception as e:
                    logger.error(f"Failed to open image {metadata.image_id}: {e}")
                    width, height = None, None

                # Update database
                self.db_manager.update_download_status(
                    image_id=metadata.image_id,
                    width=width,
                    height=height,
                    filesize=file_size
                )
                success_count += 1
            else:
                logger.warning(f"Failed to download {metadata.image_id} after {self.max_retries} attempts")
                failed_count += 1

            # Sleep between downloads (except for last image)
            if idx < total:
                self._random_sleep()

        results = DownloadResult(
            success=success_count,
            failed=failed_count,
            skipped=skipped_count
        )
        logger.info(f"Batch download complete: {results}")
        return results

    def download_pending_from_db(
            self,
            limit: Optional[int] = None,
            source: Optional[ImageSource] = None,
            label: Optional[ImageLabel] = None
    ) -> DownloadResult:
        """
        Fetch undownloaded images from database and download them.

        Args:
            limit: Maximum number of images to download (None for all)
            source: Filter by image source (optional)
            label: Filter by image label (optional)

        Returns:
            Dictionary with download stats
        """
        # Get pending downloads from database
        pending_metadata: List[ImageMetadata] = self.db_manager.get_pending_downloads(
            limit=limit if limit else 100,
            offset=0,
            source=source,
            label=label
        )

        if not pending_metadata:
            logger.info("No pending downloads found")
            return DownloadResult(
                success=0,
                failed=0,
                skipped=0
            )

        logger.info(f"Found {len(pending_metadata)} pending downloads")
        return self.download_batch(metadata_list=pending_metadata)
