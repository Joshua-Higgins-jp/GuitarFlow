from typing import List, Any, Dict, Tuple
from urllib.parse import urlencode

from loguru import logger
from requests import get, Response

from src.data_collecting.collector_database import ImageMetadataDBManager
from src.data_collecting.collectors.base_collector import BaseCollector
from src.data_collecting.collectors.collector_models_enums import ImageSource, ImageLabel, ImageMetadata
from src.utils.get_dt_now import get_dt_now_jst


class PixabayCollector(BaseCollector):
    """Pixabay API image collector"""
    def __init__(
            self,
            api_key: str,
            results_per_page: int,
            min_width: int,
            min_height: int
    ) -> None:
        super().__init__(
            api_key=api_key,
            service_name=ImageSource.PIXABAY
        )
        if not isinstance(results_per_page, int):
            raise TypeError("results_per_page must be a integer")

        if not 0 < results_per_page <= 200:
            raise ValueError("As per the API specs, Pixabay's results_per_page must be between 0 and 200")

        if not isinstance(min_width, int):
            raise TypeError("min_width must be a integer")

        if not isinstance(min_height, int):
            raise TypeError("min_height must be a integer")

        self.per_page: int = results_per_page
        self.min_width: int = min_width  # 640
        self.min_height: int = min_height  # 480
        self.base_url: str = "https://pixabay.com/api/"

    def _build_search_url(
            self,
            query: str,
            page: int
    ) -> str:
        """Build Pixabay API request URL"""
        params = {
            'key': self.api_key,
            'q': query,
            'image_type': 'photo',
            'category': 'music',
            'min_width': self.min_width,
            'min_height': self.min_height,
            'per_page': self.per_page,
            'page': page,
            'safesearch': 'true'
        }
        return f"{self.base_url}?{urlencode(params)}"

    def search(
            self,
            query: str,
            label: ImageLabel,
            max_results: int,
    ) -> List[ImageMetadata]:
        """Search Pixabay for images"""
        if not isinstance(query, str):
            raise TypeError("Query must be a string")

        total_pages: int = (max_results + self.per_page - 1) // self.per_page

        metadata_list: List[ImageMetadata] = []

        logger.info(f"Searching Pixabay: query='{query}', label={label.value}, max={max_results}")

        for page in range(1, total_pages + 1):
            url: str = self._build_search_url(
                query=query,
                page=page
            )

            try:
                response: Response = get(
                    url=url,
                    timeout=30
                )
                response.raise_for_status()
                data: Dict[str, Any] = response.json()

                hits = data.get('hits', [])
                logger.info(f"Page {page}: Found {len(hits)} images")

                for hit in hits:
                    if len(metadata_list) >= max_results:
                        break

                    metadata = ImageMetadata(
                        source=ImageSource.PIXABAY,
                        label=label.value,
                        image_id=str(hit['id']),
                        image_url=hit['largeImageURL'],  # 1280px max
                        search_query=query,
                        searched_at=get_dt_now_jst(),
                        license="Pixabay License",
                        width=hit.get('imageWidth'),
                        height=hit.get('imageHeight'),
                        filesize=hit.get('imageSize'),
                        is_downloaded=False,
                        downloaded_at=None
                    )
                    metadata_list.append(metadata)

                if len(metadata_list) >= max_results:
                    break

                # Check if we've exhausted results
                if len(hits) < self.per_page:
                    logger.info("No more results available")
                    break

            except Exception as e:
                logger.error(f"Error fetching page {page}: {e}")
                break

        logger.info(f"Found {len(metadata_list)} images for '{query}'")
        return metadata_list

    def query_and_update_database(
            self,
            database_manager: ImageMetadataDBManager,
            queries_and_labels: List[Tuple[str, ImageLabel]],
    ) -> None:
        """Query Pixabay for images"""
        total_collected: int = 0
        for query, label in queries_and_labels:
            logger.info(f"Searching Pixabay for: '{query}' (label: {label.value})")

            try:
                # Each query can fetch up to ~500 results (API limitation)
                pixabay_search_metadata: List[ImageMetadata] = self.search(
                    query=query,
                    label=label,
                    max_results=500  # 500 is pixabay's max results per query.
                )

                logger.info(f"Found {len(pixabay_search_metadata)} images for '{query}'")

                if pixabay_search_metadata:
                    db_stats_insert_metadata: Dict[str, int] = database_manager.insert_batch(metadata_list=pixabay_search_metadata)
                    logger.info(f"Database insert result: {db_stats_insert_metadata}")
                    total_collected += db_stats_insert_metadata['inserted']

            except Exception as e:
                logger.error(f"Error collecting from Pixabay for query '{query}': {e}")
                continue

        logger.info(f"Pixabay collection complete. Total new images: {total_collected}")
