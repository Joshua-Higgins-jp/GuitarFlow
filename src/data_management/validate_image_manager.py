from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image
from loguru import logger

from common.load_image import LoadImage
from data_management.validate_dataset_dir_manager import TargetsAndNonTargetsSets, DatasetValidationStage
from data_management.image_reference import DatasetFileReference


@dataclass(frozen=False)
class ImageCheckResult:
    file_ref: DatasetFileReference
    failed_reasons: list[str] = field(default_factory=list)

    @property
    def has_failed_reasons(self) -> bool:
        return len(self.failed_reasons) > 0

    @property
    def num_failed_reasons(self) -> int:
        return len(self.failed_reasons)

    @property
    def join_failed_reasons(self) -> str:
        if self.has_failed_reasons:
            return ", ".join(self.failed_reasons)
        else:
            return "No failed reasons"

    def append_fail_reason(self, stage: DatasetValidationStage, reason: str) -> None:
        self.failed_reasons.append(f"{stage.value}: {reason}")

    def log_results(self) -> None:
        if self.has_failed_reasons:
            logger.error(
                f"{self.file_ref.filename} "
                f"has failed {self.num_failed_reasons} image checks: "
                f"{self.join_failed_reasons}"
            )
        else:
            logger.info(f"{self.file_ref.filename} has no failed image checks!")


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

        To add a check, write a static method named check_<name> that takes a CheckContext and returns a reason
        string (or None if the image passes), then add it to the tuple of checks below.

        Args:
            dataset_targets_sets: The targets and non targets found when the directory was analysed.
            config: Thresholds for the checks.
        """
        self._dataset_targets_sets: TargetsAndNonTargetsSets = dataset_targets_sets

        self.targets_set: frozenset[DatasetFileReference] = self._dataset_targets_sets.targets_set
        self.non_targets_set: frozenset[DatasetFileReference] = self._dataset_targets_sets.non_targets_set

        self.config: ImageCheckConfig = config

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

    @staticmethod
    def check_min_size(ctx: CheckContext) -> Optional[str]:
        """Fail images with either side smaller than the training input size.

        Args:
            ctx: Everything known about the image.

        Returns:
            A reason string if the image is too small, otherwise None.
        """
        width, height = ctx.loaded.pixels.size
        if min(width, height) < ctx.config.img_size:
            return f"{width}x{height}, shortest side is below IMG_SIZE {ctx.config.img_size}"
        return None

    @staticmethod
    def check_aspect_ratio(ctx: CheckContext) -> Optional[str]:
        """Fail images whose long side is too many times their short side.

        Args:
            ctx: Everything known about the image.

        Returns:
            A reason string if the aspect ratio is too extreme, otherwise None.
        """
        width, height = ctx.loaded.pixels.size
        ratio = max(width, height) / min(width, height)
        if ratio > ctx.config.max_aspect_ratio:
            return f"{ratio:.2f} is above {ctx.config.max_aspect_ratio} ({width}x{height})"
        return None

    @staticmethod
    def check_animated(ctx: CheckContext) -> Optional[str]:
        """Fail files that hold an animation.

        Args:
            ctx: Everything known about the image.

        Returns:
            A reason string if the file is an animated image, otherwise None.
        """
        if ctx.loaded.format in _ANIMATION_FORMATS and ctx.loaded.n_frames > 1:
            return f"{ctx.loaded.format} with {ctx.loaded.n_frames} frames"
        return None

    @staticmethod
    def check_transparency(ctx: CheckContext) -> Optional[str]:
        """Fail images with an alpha channel or transparency key, whose hidden pixels become arbitrary in RGB.

        Args:
            ctx: Everything known about the image.

        Returns:
            A reason string if the image has transparency, otherwise None.
        """
        if ctx.loaded.has_transparency:
            return f"image has a transparency channel (mode {ctx.loaded.mode})"
        return None

    @staticmethod
    def check_supported_mode(ctx: CheckContext) -> Optional[str]:
        """Fail modes that hold more than 8 bits per pixel, which convert badly to RGB.

        Args:
            ctx: Everything known about the image.

        Returns:
            A reason string if the mode is unsupported, otherwise None.
        """
        if ctx.loaded.mode in {"I;16", "I", "F"}:
            return f"mode {ctx.loaded.mode} holds more than 8 bits per pixel"
        return None

    @staticmethod
    def check_format_matches_extension(ctx: CheckContext) -> Optional[str]:
        """Fail files whose extension does not match the format PIL detected from the content.

        Args:
            ctx: Everything known about the image.

        Returns:
            A "currently X, should be Y" message if they disagree, otherwise None.
        """
        detected = "JPEG" if ctx.loaded.format == "MPO" else ctx.loaded.format  # MPO files are ordinary .jpg photos
        if detected is None:
            return None

        valid_extensions = {ext for ext, name in Image.registered_extensions().items() if name == detected}
        if not valid_extensions or ctx.path.suffix.lower() in valid_extensions:
            return None

        should_be = _PREFERRED_EXTENSIONS.get(detected, sorted(valid_extensions)[0])
        return f"currently '{ctx.path.suffix}', should be '{should_be}' (content is {detected})"

    # ------------------------------------------------------------------ checks on the analysis array

    @staticmethod
    def check_finite(ctx: CheckContext) -> Optional[str]:
        """Guard against an empty array or NaN and Inf in float arrays. Cannot trigger for 8-bit images.

        Args:
            ctx: Everything known about the image.

        Returns:
            A reason string if the array is empty or has non-finite values, otherwise None.
        """
        arr = ctx.analysis_array
        if arr.size == 0:
            return "empty pixel array"
        if np.issubdtype(arr.dtype, np.floating) and not np.isfinite(arr).all():
            return "NaN or Inf pixel values"
        return None

    @staticmethod
    def check_flat(ctx: CheckContext) -> Optional[str]:
        """Fail solid or near-solid images.

        Args:
            ctx: Everything known about the image.

        Returns:
            A reason string if the standard deviation of pixel values is too low, otherwise None.
        """
        std = float(ctx.analysis_array.std())
        if std < ctx.config.min_std:
            return f"std {std:.1f} is below {ctx.config.min_std}"
        return None

    @staticmethod
    def check_robust_range(ctx: CheckContext) -> Optional[str]:
        """Fail images with almost no spread between dark and light pixels, ignoring stray extreme pixels.

        Args:
            ctx: Everything known about the image.

        Returns:
            A reason string if the spread between the 1st and 99th percentile is too small, otherwise None.
        """
        low, high = np.percentile(ctx.analysis_array, [1, 99])
        if high - low < ctx.config.min_robust_range:
            return f"{high - low:.1f} between 1st and 99th percentile, below {ctx.config.min_robust_range}"
        return None

    @staticmethod
    def check_dominant_colour(ctx: CheckContext) -> Optional[str]:
        """Fail images where almost every pixel is the same colour.

        Colours are quantised to 16 levels per channel before counting.

        Args:
            ctx: Everything known about the image.

        Returns:
            A reason string if one colour dominates, otherwise None.
        """
        quantised = (ctx.analysis_array >> 4).astype(np.uint16)
        codes = (quantised[..., 0] << 8) | (quantised[..., 1] << 4) | quantised[..., 2]
        dominant = float(np.bincount(codes.ravel()).max() / codes.size)
        if dominant > ctx.config.max_dominant_fraction:
            return f"{dominant:.1%} of pixels are one colour"
        return None

    # ------------------------------------------------------------------ orchestration

    @staticmethod
    def _run_checks(checks: tuple[CheckFn, ...], context: CheckContext) -> tuple[CheckFailure, ...]:
        """Run each check and collect the ones that failed.

        Args:
            checks: Check functions to run, each taking the context and returning a reason or None.
            context: Everything known about the image.

        Returns:
            One CheckFailure per failed check, in the order the checks ran.
        """
        failures: list[CheckFailure] = []
        for check in checks:
            reason = check(context)
            if reason is not None:
                failures.append(CheckFailure(check.__name__.removeprefix("check_"), reason))
        return tuple(failures)

    def check_image(self, reference: ImageReference) -> ImageReport:
        """Open one image once and run every check on it, collecting failures along the way.

        Nothing is moved or written. An undecodable file gets a single "decode" failure and no further checks. The
        pixel hash is only computed for images that pass every check.

        Args:
            reference: The image to check.

        Returns:
            A report with the failed checks, and the pixel hash if the image passed everything.
        """
        path: Path = reference.resolved_filepath

        try:
            loaded = LoadImage.from_path(path)
        except Exception as error:  # any decode failure means the file is unusable for training
            failure = CheckFailure("decode", f"could not be decoded ({type(error).__name__}: {error})")
            return ImageReport(reference, (failure,), None, None)

        context = CheckContext(
            path=path,
            loaded=loaded,
            analysis_array=self.to_analysis_array(loaded.pixels, self.config.analysis_size),
            config=self.config,
        )
        failures = self._run_checks(self._checks, context)

        digest = None if failures else loaded.pixel_sha256
        return ImageReport(reference, failures, digest, loaded.exif_orientation)
