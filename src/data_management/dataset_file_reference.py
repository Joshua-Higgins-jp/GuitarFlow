from dataclasses import dataclass
from pathlib import Path

from common.constants.accepted_image_formats import AcceptedImageFormats
from common.constants.class_labels import ClassLabels
from common.constants.source_labels import SourceLabels


@dataclass(frozen=True)
class DatasetFileReference:
    """
    creates a file reference object for each file found in the raw image data directory

    these are not necessarily images.. but we will need to perform some logic later
    if image
    """
    images_raw_dir: Path
    label: ClassLabels
    source: SourceLabels
    filename: str

    @property
    def resolved_filepath(self) -> Path:
        """returns the resolved filepath for the image on disk"""
        return Path(self.images_raw_dir / f"{self.label.value}/{self.source.value}/{self.filename}")

    @property
    def filetype_raw(self) -> str:
        """not converted to the accepted image format enum class, just the raw value"""
        return self.resolved_filepath.suffix

    @property
    def is_target_image_filetype(self) -> bool:
        """
        a naive determination of a files filetype based on the de facto file extension.
        if a file reference has an extension in AcceptedImageFormats, then we return True, else False.
        """
        if self.filetype_raw in AcceptedImageFormats.as_tuple(include_dot=True):
            return True
        return False
