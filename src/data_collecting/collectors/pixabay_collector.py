from typing import List
from urllib.parse import urlencode

from loguru import logger
from requests import get, Response

from src.data_collecting.collectors.base_collector import BaseCollector
from src.data_collecting.collectors.collector_models import ImageSource, ImageLabel, ImageMetadata
from src.utils.get_dt_now import get_dt_now


class PixabayCollector(BaseCollector):
    """Pixabay API image collector"""

    # should be ClassVar[str]
    BASE_URL = "https://pixabay.com/api/"

    def __init__(self, api_key: str):
        super().__init__(
            api_key=api_key,
            service_name=ImageSource.PIXABAY.value
        )

    def search(
            self,
            query: str,
            label: ImageLabel,
            max_results: int = 100,
            min_width: int = 640,
            min_height: int = 480
    ) -> List[ImageMetadata]:
        """Search Pixabay for images"""
        metadata_list: List[ImageMetadata] = []
        per_page: int = 200  # Max allowed by API
        total_pages: int = (max_results + per_page - 1) // per_page

        logger.info(f"Searching Pixabay: query='{query}', label={label.value}, max={max_results}")

        for page in range(1, total_pages + 1):
            url: str = self._build_search_url(
                query=query,
                page=page,
                per_page=per_page,
                min_width=min_width,
                min_height=min_height
            )

            try:
                response: Response = get(url, timeout=30)
                response.raise_for_status()
                data = response.json()

                hits = data.get('hits', [])
                logger.info(f"Page {page}: Found {len(hits)} images")

                for hit in hits:
                    if len(metadata_list) >= max_results:
                        break

                    metadata = ImageMetadata(
                        source=ImageSource.PIXABAY,
                        label=label,
                        source_id=str(hit['id']),
                        url=hit['largeImageURL'],  # 1280px max
                        search_query=query,
                        license="Pixabay License",
                        downloaded_at=get_dt_now(),
                        width=hit.get('imageWidth'),
                        height=hit.get('imageHeight'),
                        filesize=hit.get('imageSize')
                    )
                    metadata_list.append(metadata)

                if len(metadata_list) >= max_results:
                    break

                # Check if we've exhausted results
                if len(hits) < per_page:
                    logger.info("No more results available")
                    break

            except Exception as e:
                logger.error(f"Error fetching page {page}: {e}")
                break

        logger.info(f"Found {len(metadata_list)} images for '{query}'")
        return metadata_list

    def _build_search_url(
            self,
            query: str,
            page: int,
            per_page: int = 200,
            min_width: int = 640,
            min_height: int = 480
    ) -> str:
        """Build Pixabay API request URL"""
        params = {
            'key': self.api_key,
            'q': query,
            'image_type': 'photo',
            'min_width': min_width,
            'min_height': min_height,
            'per_page': per_page,
            'page': page,
            'safesearch': 'true'
        }
        return f"{self.BASE_URL}?{urlencode(params)}"
