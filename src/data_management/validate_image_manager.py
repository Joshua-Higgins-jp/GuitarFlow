from dataclasses import dataclass, field
from typing import Callable, Optional, Union

import numpy as np
from PIL import Image
from loguru import logger

from common.load_image import LoadImage
from data_management.dataset_file_reference import DatasetFileReference
from data_management.validate_dataset_dir_manager import TargetsAndNonTargetsSets, DatasetValidationStage

_ANIMATION_FORMATS: frozenset[str] = frozenset({"GIF", "PNG", "WEBP"})  # PNG covers APNG; add "TIFF" to also reject multi-page TIFFs

_PREFERRED_EXTENSIONS: dict[str, str] = {
    "JPEG": ".jpg",
    "PNG": ".png",
    "GIF": ".gif",
    "WEBP": ".webp",
    "BMP": ".bmp",
    "TIFF": ".tiff",
    "JPEG2000": ".jp2",
}


@dataclass(frozen=True)
class ImageCheckConfig:
    """
    Thresholds for the dataset image checks.

    Attributes:
        img_size: Training input size. Images with a shorter side below this fail.
        max_aspect_ratio: Longest side divided by shortest side, above which an image fails.
        min_std: Minimum standard deviation of pixel values (0 to 255 scale), below which an image is flat.
        min_robust_range: Minimum gap between the 1st and 99th percentile of pixel values.
        max_dominant_fraction: Fraction of pixels (0 to 1) sharing one quantised colour, above which an image fails.
        analysis_size: Longest side in pixels of the downscaled copy used for the statistics checks.
    """
    img_size: int = 224
    max_aspect_ratio: float = 4.0
    min_std: float = 5.0
    min_robust_range: float = 20.0
    max_dominant_fraction: float = 0.95
    analysis_size: int = 256


@dataclass(frozen=False, eq=False)  # eq=False keeps identity hashing, so results can live in a set
class ImageCheckResult:
    file_ref: DatasetFileReference
    failed_reasons: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return len(self.failed_reasons) == 0

    @property
    def num_failed_reasons(self) -> int:
        return len(self.failed_reasons)

    @property
    def join_failed_reasons(self) -> str:
        if not self.passed:
            return ", ".join(self.failed_reasons)
        else:
            return "PASSED"

    def append_fail_reason(self, stage: DatasetValidationStage, reason: str) -> None:
        self.failed_reasons.append(f"{stage.value}: {reason}")

    def log_results(self) -> None:
        if not self.passed:
            logger.error(
                f"FAILED {self.num_failed_reasons} CHECKS: "
                f"{self.file_ref.filename}: "
                f"{self.join_failed_reasons}"
            )
        else:
            logger.info(f"PASSED ALL CHECKS: {self.file_ref.filename}")


@dataclass(frozen=True)
class ImageChecksResultsSets:
    """
    docstrnng
    """
    passed: frozenset[ImageCheckResult]
    failed: frozenset[ImageCheckResult]

    @property
    def num_passed(self) -> int:
        return len(self.passed)

    @property
    def num_failed(self) -> int:
        return len(self.failed)

    @property
    def total_files(self) -> int:
        return self.num_passed + self.num_failed

    def log_results(self) -> None:
        logger.info(
            f"Total Files = {self.total_files}; "
            f"Passed={self.num_passed}; "
            f"Failed={self.num_failed}"
        )


class DatasetValidationImageCheckManager:
    def __init__(
            self,
            dataset_targets_sets: TargetsAndNonTargetsSets,
            config: ImageCheckConfig,
    ) -> None:
        """
        Check the target images of an analysed dataset directory.

        Now that the directory has been analysed we have targets and non targets:
         - the non targets go straight to Quarantine (handled elsewhere)
         - each target is loaded once and run through every check, and each failed check is recorded
         - a target with at least one failed check is sent to Quarantine (handled elsewhere)

        To add a check, write a method named check_<name> that returns a reason string (or None if the image passes).
        If it needs the loaded image, take a LoadImage and add it to self._image_checks. If it only needs pixel
        statistics, take the downscaled uint8 array and add it to self._array_checks.

        Args:
            dataset_targets_sets: The targets and non targets found when the directory was analysed.
            config: Thresholds for the checks.
        """
        self._dataset_targets_sets: TargetsAndNonTargetsSets = dataset_targets_sets

        self.targets_set: frozenset[DatasetFileReference] = self._dataset_targets_sets.targets_set
        self.non_targets_set: frozenset[DatasetFileReference] = self._dataset_targets_sets.non_targets_set

        self.config: ImageCheckConfig = config

        self._image_checks: tuple[Callable[[LoadImage], Optional[str]], ...] = (
            self.check_min_size,
            self.check_aspect_ratio,
            self.check_animated,
            self.check_transparency,
            self.check_supported_mode,
            self.check_format_matches_extension,
        )
        self._array_checks: tuple[Callable[[np.ndarray], Optional[str]], ...] = (
            self.check_finite,
            self.check_flat,
            self.check_robust_range,
            self.check_dominant_colour,
        )

    @staticmethod
    def to_analysis_array(pixels: Image.Image, size: int) -> np.ndarray:
        """
        Make a small uint8 array for the pixel statistics checks, leaving the original image untouched.

        The statistics checks only need the overall look of the image, so they run on a downscaled copy. This is faster
        and keeps thresholds comparable between small and very large images. Images already smaller than the target
        are not enlarged.

        Args:
            pixels: Decoded RGB image.
            size: Longest side of the downscaled copy in pixels.

        Returns:
            A uint8 array of shape (H, W, 3).
        """
        downscaled = pixels.copy()  # thumbnail() resizes in place, so work on a copy
        downscaled.thumbnail((size, size))
        return np.asarray(downscaled)

    # ------------------------------------------------------------------ checks on the loaded image

    def check_min_size(self, load_image: LoadImage) -> Optional[str]:
        """Fail images with either side smaller than the training input size.

        Args:
            load_image: The loaded image.

        Returns:
            A reason string if the image is too small, otherwise None.
        """
        width, height = load_image.pixels.size
        if min(width, height) < self.config.img_size:
            return f"{width}x{height}, shortest side is below IMG_SIZE {self.config.img_size}"
        return None

    def check_aspect_ratio(self, load_image: LoadImage) -> Optional[str]:
        """Fail images whose long side is too many times their short side.

        Args:
            load_image: The loaded image.

        Returns:
            A reason string if the aspect ratio is too extreme, otherwise None.
        """
        width, height = load_image.pixels.size
        ratio = max(width, height) / min(width, height)
        if ratio > self.config.max_aspect_ratio:
            return f"{ratio:.2f} is above {self.config.max_aspect_ratio} ({width}x{height})"
        return None

    @staticmethod
    def check_animated(load_image: LoadImage) -> Optional[str]:
        """Fail files that hold an animation.

        Args:
            load_image: The loaded image.

        Returns:
            A reason string if the file is an animated image, otherwise None.
        """
        if load_image.format in _ANIMATION_FORMATS and load_image.n_frames > 1:
            return f"{load_image.format} with {load_image.n_frames} frames"
        return None

    @staticmethod
    def check_transparency(load_image: LoadImage) -> Optional[str]:
        """Fail images with an alpha channel or transparency key, whose hidden pixels become arbitrary in RGB.

        Args:
            load_image: The loaded image.

        Returns:
            A reason string if the image has transparency, otherwise None.
        """
        if load_image.has_transparency:
            return f"image has a transparency channel (mode {load_image.mode})"
        return None

    @staticmethod
    def check_supported_mode(load_image: LoadImage) -> Optional[str]:
        """Fail modes that hold more than 8 bits per pixel, which convert badly to RGB.

        Args:
            load_image: The loaded image.

        Returns:
            A reason string if the mode is unsupported, otherwise None.
        """
        if load_image.mode in {"I;16", "I", "F"}:
            return f"mode {load_image.mode} holds more than 8 bits per pixel"
        return None

    @staticmethod
    def check_format_matches_extension(load_image: LoadImage) -> Optional[str]:
        """Fail files whose extension does not match the format PIL detected from the content.

        Args:
            load_image: The loaded image.

        Returns:
            A "currently X, should be Y" message if they disagree, otherwise None.
        """
        detected = "JPEG" if load_image.format == "MPO" else load_image.format  # MPO files are ordinary .jpg photos
        if detected is None:
            return None

        valid_extensions = {ext for ext, name in Image.registered_extensions().items() if name == detected}
        suffix = load_image.filepath.suffix
        if not valid_extensions or suffix.lower() in valid_extensions:
            return None

        should_be = _PREFERRED_EXTENSIONS.get(detected, sorted(valid_extensions)[0])
        return f"currently '{suffix}', should be '{should_be}' (content is {detected})"

    # ------------------------------------------------------------------ checks on the analysis array

    @staticmethod
    def check_finite(analysis_array: np.ndarray) -> Optional[str]:
        """Guard against an empty array or NaN and Inf in float arrays. Cannot trigger for 8-bit images.

        Args:
            analysis_array: The downscaled pixel array.

        Returns:
            A reason string if the array is empty or has non-finite values, otherwise None.
        """
        if analysis_array.size == 0:
            return "empty pixel array"
        if np.issubdtype(analysis_array.dtype, np.floating) and not np.isfinite(analysis_array).all():
            return "NaN or Inf pixel values"
        return None

    def check_flat(self, analysis_array: np.ndarray) -> Optional[str]:
        """Fail solid or near-solid images.

        Args:
            analysis_array: The downscaled pixel array.

        Returns:
            A reason string if the standard deviation of pixel values is too low, otherwise None.
        """
        std = float(analysis_array.std())
        if std < self.config.min_std:
            return f"std {std:.1f} is below {self.config.min_std}"
        return None

    def check_robust_range(self, analysis_array: np.ndarray) -> Optional[str]:
        """Fail images with almost no spread between dark and light pixels, ignoring stray extreme pixels.

        Args:
            analysis_array: The downscaled pixel array.

        Returns:
            A reason string if the spread between the 1st and 99th percentile is too small, otherwise None.
        """
        low, high = np.percentile(analysis_array, [1, 99])
        if high - low < self.config.min_robust_range:
            return f"{high - low:.1f} between 1st and 99th percentile, below {self.config.min_robust_range}"
        return None

    def check_dominant_colour(self, analysis_array: np.ndarray) -> Optional[str]:
        """Fail images where almost every pixel is the same colour.

        Colours are quantised to 16 levels per channel before counting.

        Args:
            analysis_array: The downscaled pixel array.

        Returns:
            A reason string if one colour dominates, otherwise None.
        """
        quantised = (analysis_array >> 4).astype(np.uint16)
        codes = (quantised[..., 0] << 8) | (quantised[..., 1] << 4) | quantised[..., 2]
        dominant = float(np.bincount(codes.ravel()).max() / codes.size)
        if dominant > self.config.max_dominant_fraction:
            return f"{dominant:.1%} of pixels are one colour"
        return None

    # ------------------------------------------------------------------ running the checks

    def _check_target(self, target_image_ref: DatasetFileReference) -> ImageCheckResult:
        """Load one target image and run every check on it, recording each failure.

        Args:
            target_image_ref: The target image to check.

        Returns:
            The result holding every failed reason, which is empty if the image passed.
        """
        result = ImageCheckResult(file_ref=target_image_ref)

        try:
            load_image: LoadImage = LoadImage.from_path(filepath=target_image_ref.resolved_filepath)

        except Exception as e:
            logger.error(f"Couldn't load target image {target_image_ref.resolved_filepath}: {e}")
            result.append_fail_reason(
                stage=DatasetValidationStage.IMAGE,
                reason=f"Couldn't open image, cause={e}"
            )
            return result  # nothing else can run without a loaded image

        for check in self._image_checks:
            self._run_one_check(
                result=result,
                check=check,
                check_input=load_image
            )

        try:
            analysis_array = self.to_analysis_array(
                pixels=load_image.pixels,
                size=self.config.analysis_size
            )

        except Exception as e:
            logger.error(f"Couldn't make analysis array {target_image_ref.resolved_filepath}: {e}")
            result.append_fail_reason(
                stage=DatasetValidationStage.IMAGE,
                reason=f"Couldn't make analysis array, cause={e}",
            )
            return result

        for check in self._array_checks:
            self._run_one_check(
                result=result,
                check=check,
                check_input=analysis_array
            )

        return result

    @staticmethod
    def _run_one_check(
            result: ImageCheckResult,
            check: Callable[..., Optional[str]],
            check_input: Union[LoadImage, np.ndarray]
    ) -> None:
        """Run a single check and record its reason on the result if it fails or crashes.

        Args:
            result: The result to append any failed reason to.
            check: The check method to run.
            check_input: What the check takes, either a LoadImage or an analysis array.
        """
        name = check.__name__.removeprefix("check_")
        try:
            reason = check(check_input)

        except Exception as e:
            result.append_fail_reason(
                stage=DatasetValidationStage.IMAGE,
                reason=f"{name}: check crashed, cause={e}"
            )
            return

        if reason is not None:
            result.append_fail_reason(
                stage=DatasetValidationStage.IMAGE,
                reason=f"{name}: {reason}"
            )

    # TODO: implement  the check_duplicates_via_image_hash from LoadImage.hash and compare to all other images??


    def run_checks(self) -> ImageChecksResultsSets:
        """
        Run every check on every target, and mark every non target as not a target.

        Returns:
            The results split into passed (targets with no failed checks) and failed (failed targets plus all non
            targets).
        """
        passed: set[ImageCheckResult] = set()
        failed: set[ImageCheckResult] = set()

        # Some targets can fail checks, so they move to failed
        for target_image_ref in self.targets_set:
            check_result = self._check_target(target_image_ref=target_image_ref)
            check_result.log_results()
            if check_result.passed:
                passed.add(check_result)
            else:
                failed.add(check_result)

        # Non targets come from the failed dir, so they fail by default
        for non_target_ref in self.non_targets_set:
            non_target_check = ImageCheckResult(file_ref=non_target_ref)
            non_target_check.append_fail_reason(stage=DatasetValidationStage.IMAGE, reason="Not a target")
            non_target_check.log_results()
            failed.add(non_target_check)

        return ImageChecksResultsSets(
            passed=frozenset(passed),
            failed=frozenset(failed)
        )
