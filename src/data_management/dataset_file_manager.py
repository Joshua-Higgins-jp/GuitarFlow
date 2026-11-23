from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Iterator, Optional

from PIL import Image

from loguru import logger

from common.constants.accepted_image_formats import AcceptedImageFormats
from common.constants.class_labels import ClassLabels
from common.constants.source_labels import SourceLabels
from common.utils.rename_file_with_lowercase_extension import rename_file_with_lowercase_extension
from data_management.image_reference import ImageDatasetFileReference


class DatasetValidationStage(StrEnum):
    FILESYSTEM__DIR_STRUCTURE_CHECK = "filesystem__dir_structure_check"
    FILESYSTEM__FILETYPES_CHECK = "filesystem__filetypes_check"

    IMAGE = "image"
    DUPLICATE = "duplicate"


@dataclass(frozen=True)
class UnexpectedDirEvent:
    directory: Path
    reason: str


@dataclass(frozen=True)
class ValidateDirStructureResult:
    results: list[UnexpectedDirEvent]

    @property
    def has_problems(self) -> bool:
        return self.num_problems > 0

    @property
    def num_problems(self) -> int:
        return len(self.results)

    def log_results(self, relative_to_dir: Path) -> None:
        for result in self.results:
            logger.warning(f"Directory `{result.directory.relative_to(relative_to_dir)}` is {result.reason}")

    def runtime_error_str(self) -> str:
        return (
            f"{self.num_problems} unexpected directory entries found. "
            f"Check logs and rectify, then try again."
        )


@dataclass(frozen=True)
class TargetsAndNonTargetsSets:
    targets_set: frozenset[ImageDatasetFileReference]
    non_targets_set: frozenset[ImageDatasetFileReference]

    @property
    def num_targets(self) -> int:
        return len(self.targets_set)

    @property
    def num_non_targets(self) -> int:
        return len(self.non_targets_set)

    @property
    def total_files(self) -> int:
        return self.num_targets + self.num_non_targets

    def log_results(self) -> None:
        logger.info(
            f"Total non-system files in dataset = {self.total_files}; "
            f"targets={self.num_targets}; "
            f"non-targets={self.num_non_targets}"
        )


@dataclass(frozen=False)
class QuarantineEntry:
    """
    One flagged file.

    Attributes:
        file_ref: The file that should be quarantined.
        reasons: Why the file was quarantined.

        frozen=False because we append reasons
    """
    file_ref: ImageDatasetFileReference
    reasons: list[str] = field(default_factory=list)

    def append_reason(self, stage: DatasetValidationStage, reason: str) -> None:
        self.reasons.append(f"{stage.value}: {reason}")


@dataclass(frozen=True)
class QuarantineReport:
    """
    What one stage found. Returned by every check loop and never acted on by the stage itself.

    Attributes:
        stage: Which stage produced the report.
        entries: The files it wants quarantined.
    """
    stage: DatasetValidationStage
    entries: tuple[QuarantineEntry, ...] = ()


class DatasetValidationDirFilesManager:
    # Finder recreates these constantly, so they are ignored rather than quarantined.
    IGNORED_FILENAMES: frozenset[str] = frozenset({".DS_Store"})  # TODO: what about other file systems like windows and linux? add thumbs.db LOLOL

    def __init__(
            self,
            images_raw_dir: Path,
    ) -> None:
        """
        DatasetDirFilesManager has 2 jobs:
        1. Validate the directory structure of the dataset
            - Ensure expected dirs exist in their expected structure
            - immediately stop the dataset ingestion pipeline if there are unexpected directories
              and let the user handle it (may involve manually moving data to correct location)

        2. Validate that only approved system files and accepted image filetypes exist in the expected directory
            structure from (1).
            - Anything that is (a) not an approved system file or (b) not an approved image file, is
              documented in a QuarentineEntry and these are collected into a QuarantineReport.

        Usage:
        Step 1:
            Init the object

        Step 2:
            Run validate_dir_structure() -> ValidateDirStructureResult

        Gate:
            if validate_dir_structure().has_problems: raise RuntimeError(f"unable to continue due to directory issues... {e}")

        Step 3:
            Run validate_filetypes() -> QuarantineReport

        The step 3 result is passed in to the QuarantineManager for handling (e.g. generating the csv manifest report, dry_run=False)
            and the actual logic for csv writing, file moving, extension lowering, etc., logic all exists there.

        Args:
            images_raw_dir: Path to the raw images directory.
        """
        self.images_raw_dir: Path = images_raw_dir

    def _create_image_raw_subdirectories(self) -> None:
        """
        Creates expected training dataset directory structure. exists ok is True
        """
        for label in ClassLabels:
            for source in SourceLabels:
                final_dir_path: Path = self.images_raw_dir / label.value / source.value
                final_dir_path.mkdir(parents=True, exist_ok=True)

    def _assert_images_raw_layout_is_valid(self) -> ValidateDirStructureResult:
        """
        if anything in the tree is not part of the expected layout, log it and raise out

        The expected layout is images_raw_dir/<label>/<source>/<files>. Anything else is flagged, because it
        would never be reached by the scan: at the root, any entry that is not a label directory; inside each
        label, any entry that is not a source directory; inside each source, any subdirectory.
        System files such as .DS_Store are ignored at every level.

        Raises:
            RuntimeError: If any unexpected entry is found. Every one is logged first.
        """
        unexpected_dir_events: list[UnexpectedDirEvent] = []

        # Root level: only label directories belong here.
        for item_path in sorted(self.images_raw_dir.iterdir()):
            if item_path.name in self.IGNORED_FILENAMES:
                continue
            if not (item_path.is_dir() and item_path.name in ClassLabels.as_set()):
                unexpected_dir_events.append(UnexpectedDirEvent(
                    directory=item_path,
                    reason=f"not an expected label directory.")
                )

        for label in ClassLabels:
            label_dir: Path = self.images_raw_dir / label.value

            # Label level: only source directories belong here.
            for item_path in sorted(label_dir.iterdir()):
                if item_path.name in self.IGNORED_FILENAMES:
                    continue
                if not (item_path.is_dir() and item_path.name in SourceLabels.as_set()):
                    unexpected_dir_events.append(UnexpectedDirEvent(
                        directory=item_path,
                        reason=f"not an expected source directory.")
                    )

            # Source level: files only, no subdirectories.
            for source in SourceLabels:
                for item_path in sorted((label_dir / source.value).iterdir()):
                    if item_path.is_dir():
                        unexpected_dir_events.append(UnexpectedDirEvent(
                            directory=item_path,
                            reason=f"a source directory should not have subdirectories.")
                        )

        # If anything is unexpected, log it and raise to stop until the user has corrected their dataset.
        return ValidateDirStructureResult(results=unexpected_dir_events)


    def validate_dir_structure(self) -> ValidateDirStructureResult:
        # 1. Make the expected dir structure
        self._create_image_raw_subdirectories()

        # 2. Check only the expected dir structure exists. log issues, return structured res object
        return self._assert_images_raw_layout_is_valid()


    def _iter_raw_files(self) -> Iterator[tuple[ClassLabels, SourceLabels, Path]]:
        """
        Yield every regular file under images_raw_dir/<label>/<source>/, skipping ignored system files.

        Missing directories are warned about and skipped. Anything that is not a regular file is warned about and
        skipped. Each directory is listed up front, so it is safe to move or rename files while iterating.

        Yields:
            A tuple of (label, source, file path), ordered by label, then source, then filename.
        """
        for label in ClassLabels:
            for source in SourceLabels:
                directory: Path = self.images_raw_dir / label.value / source.value
                if not directory.is_dir():
                    logger.warning(f"Expected directory is missing, skipping: {directory}")
                    continue

                for path in sorted(directory.iterdir()):
                    if path.name in self.IGNORED_FILENAMES:
                        continue
                    if not path.is_file():
                        logger.warning(f"Skipping entry that is not a regular file: {path}")
                        continue
                    yield label, source, path

    def _gen_dataset_all_file_refs(self) -> frozenset[ImageDatasetFileReference]:
        dataset_file_refs: set[ImageDatasetFileReference] = set()
        for label, source, path in self._iter_raw_files():
            dataset_file_refs.add(
                ImageDatasetFileReference(
                images_raw_dir=self.images_raw_dir,
                label=label,
                source=source,
                filename=path.name
                )
            )
        return frozenset(dataset_file_refs)

    def gen_target_sets(self) -> TargetsAndNonTargetsSets:
        # 0. confirm images raw directory structure is valid
        self.validate_dir_structure()

        # 1. init empty return sets
        targets_set: set[ImageDatasetFileReference] = set()
        non_targets_set: set[ImageDatasetFileReference] = set()

        # 2. generate all file refs from the image dataset
        dataset_all_file_refs: frozenset[ImageDatasetFileReference] = self._gen_dataset_all_file_refs()

        # 3. split them by target / non-target
        for file_ref in dataset_all_file_refs:
            targets_set.add(file_ref) if file_ref.is_target_image_filetype else non_targets_set.add(file_ref)

        # 4. return ground truth targets and non targets sets
        return TargetsAndNonTargetsSets(
            targets_set=frozenset(targets_set),
            non_targets_set=frozenset(non_targets_set)
        )


from common.dirs_paths import _PROJECT_DIR
from time import monotonic
start_time = monotonic()
logger.info(start_time)
dataset_dir_files_mgr = DatasetValidationDirFilesManager(images_raw_dir=_PROJECT_DIR / "data" / "images_raw")
dataset_target_sets: TargetsAndNonTargetsSets = dataset_dir_files_mgr.gen_target_sets()  # only this takes 0.021s
end_time = monotonic()
logger.info(end_time)
dataset_target_sets.log_results()
logger.info(f"time taken = {round((end_time - start_time), 3)}s")


@dataclass(frozen=False)
class ImageCheckResult:
    file_ref: ImageDatasetFileReference
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
            return ""

    def append_fail_reason(self, stage_name: DatasetValidationStage, reason: str) -> None:
        self.failed_reasons.append(reason)

    def log_results(self) -> None:
        if self.has_failed_reasons:
            logger.error(
                f"{self.file_ref.filename} "
                f"has failed {self.num_failed_reasons} image checks: "
                f"{self.join_failed_reasons}"
            )
        else:
            logger.info(f"{self.file_ref.filename} has no failed image checks!")

from numpy import asarray, ndarray
from PIL import ImageOps

@dataclass(frozen=True)
class LoadedImage:
    """Everything the checks need, gathered from one open of the file.

    Attributes:
        pixels: Fully decoded RGB image with the EXIF orientation already applied.
        format: Format PIL detected from the file content, for example "JPEG". None if unknown.
        n_frames: Number of frames in the file (1 for an ordinary still image).
        transparent_fraction: Share of pixels that are not fully opaque, measured before conversion to RGB.
        exif_orientation: Orientation tag stored in the file (1 if missing). The file itself is never changed.
    """
    pixels: Image.Image
    format: Optional[str]
    n_frames: int
    transparent_fraction: float
    exif_orientation: int


_EXIF_ORIENTATION_TAG: int = 0x0112


class DatasetValidationImageCheckManager:
    def __init__(
            self,
            dataset_targets_sets: TargetsAndNonTargetsSets
    ) -> None:
        """
        Now that we've analysed the directory, we have targets and non targets

        Simply put:
         - the non targets will immediately go to Quarantine
         - the targets will be run through image checks and each check will append a failed result
         - if a target has a non-zero failed result list, it gets sent to Quarantine.

        Args:
            dataset_targets_sets:
        """
        self._dataset_targets_sets: TargetsAndNonTargetsSets = dataset_targets_sets

        self.targets_set: frozenset[ImageDatasetFileReference] = self._dataset_targets_sets.targets_set
        self.non_targets_set: frozenset[ImageDatasetFileReference] = self._dataset_targets_sets.non_targets_set

    @staticmethod
    def load_image(path: Path) -> LoadedImage:
        """Open a file once, decode it fully, and gather everything the checks need.

        The EXIF orientation is applied in memory only. The file on disk is never changed. Decompression bomb
        warnings are treated as errors.

        Args:
            path: Path of the image file.

        Returns:
            A LoadedImage holding the upright RGB pixels and the facts that conversion to RGB would lose.

        Raises:
            Exception: Whatever PIL raises for a file it cannot fully decode (truncated, corrupt, wrong format,
                decompression bomb). The caller treats any failure as an undecodable file.
        """
        with Image.open(path) as image:
            image.load()  # forces a complete decode, so truncated files fail here

            orientation = image.getexif().get(_EXIF_ORIENTATION_TAG, 1)
            transparent_fraction = 0.0
            if "A" in image.getbands() or "transparency" in image.info:
                alpha: ndarray = asarray(image.convert("RGBA").getchannel("A"))
                transparent_fraction = float((alpha < 255).mean())

            return LoadedImage(
                pixels=ImageOps.exif_transpose(image).convert("RGB"),
                format=image.format,
                n_frames=getattr(image, "n_frames", 1),
                transparent_fraction=transparent_fraction,
                exif_orientation=orientation if isinstance(orientation, int) else 1,
            )

    @staticmethod
    def to_thumbnail_array(pixels: Image.Image, size: int) -> np.ndarray:
        """Make a small uint8 array for the pixel statistics, leaving the original image untouched.

        Args:
            pixels: Decoded RGB image.
            size: Longest side of the thumbnail in pixels.

        Returns:
            A uint8 array of shape (H, W, 3).
        """
        thumbnail = pixels.copy()  # thumbnail() resizes in place, so work on a copy
        thumbnail.thumbnail((size, size))
        return np.asarray(thumbnail)

    # ------------------------------------------------------------------ checks on the loaded image

    @staticmethod
    def check_min_size(pixels: Image.Image, img_size: int) -> Optional[str]:
        """Reject images with either side smaller than the training input size.

        Args:
            pixels: Decoded upright RGB image.
            img_size: Training input size in pixels.

        Returns:
            A reason string if the image is too small, otherwise None.
        """
        width, height = pixels.size
        if min(width, height) < img_size:
            return f"too small: {width}x{height}, shortest side is below IMG_SIZE {img_size}"
        return None

    @staticmethod
    def check_aspect_ratio(pixels: Image.Image, max_ratio: float) -> Optional[str]:
        """Reject images whose long side is too many times their short side.

        Args:
            pixels: Decoded upright RGB image.
            max_ratio: Largest allowed long side divided by short side.

        Returns:
            A reason string if the aspect ratio is too extreme, otherwise None.
        """
        width, height = pixels.size
        ratio = max(width, height) / min(width, height)
        if ratio > max_ratio:
            return f"aspect ratio {ratio:.2f} is above {max_ratio} ({width}x{height})"
        return None

    @staticmethod
    def check_animated(loaded: LoadedImage) -> Optional[str]:
        """Reject files that hold an animation.

        Args:
            loaded: The loaded image.

        Returns:
            A reason string if the file is an animated image, otherwise None.
        """
        if loaded.format in _ANIMATION_FORMATS and loaded.n_frames > 1:
            return f"animated image ({loaded.n_frames} frames)"
        return None

    @staticmethod
    def check_transparency(loaded: LoadedImage, max_fraction: float) -> Optional[str]:
        """Reject images with too many transparent pixels, which turn black or arbitrary when converted to RGB.

        Args:
            loaded: The loaded image.
            max_fraction: Largest allowed share of not fully opaque pixels.

        Returns:
            A reason string if too much of the image is transparent, otherwise None.
        """
        if loaded.transparent_fraction > max_fraction:
            return f"{loaded.transparent_fraction:.1%} of pixels are transparent"
        return None

    @staticmethod
    def check_format_matches_extension(loaded: LoadedImage, path: Path) -> Optional[str]:
        """Report when the file extension does not match the format PIL detected from the content.

        Args:
            loaded: The loaded image.
            path: Path of the file, used for its extension.

        Returns:
            A "currently X, should be Y" message if they disagree, otherwise None.
        """
        detected = "JPEG" if loaded.format == "MPO" else loaded.format  # MPO files are ordinary .jpg photos
        if detected is None:
            return None

        valid_extensions = {ext for ext, name in Image.registered_extensions().items() if name == detected}
        if not valid_extensions or path.suffix.lower() in valid_extensions:
            return None

        should_be = _PREFERRED_EXTENSIONS.get(detected, sorted(valid_extensions)[0])
        return f"format mismatch: currently '{path.suffix}', should be '{should_be}' (content is {detected})"

    @staticmethod
    def pixel_sha256(pixels: Image.Image) -> str:
        """Hash the image dimensions and RGB pixel values, ignoring how the file stores them.

        Args:
            pixels: Decoded upright RGB image at full resolution (not a thumbnail).

        Returns:
            A 64 character lowercase hex digest.
        """
        rgb = pixels if pixels.mode == "RGB" else pixels.convert("RGB")
        digest = hashlib.sha256()
        digest.update(f"{rgb.width}x{rgb.height}:RGB:".encode("ascii"))
        digest.update(rgb.tobytes())
        return digest.hexdigest()

    # ------------------------------------------------------------------ checks on the thumbnail array

    @staticmethod
    def check_finite(arr: np.ndarray) -> Optional[str]:
        """Guard against NaN or Inf in float arrays. Cannot trigger for arrays decoded from 8-bit images.

        Args:
            arr: Pixel array.

        Returns:
            A reason string if non-finite values are present, otherwise None.
        """
        if arr.size == 0:
            return "empty pixel array"
        if np.issubdtype(arr.dtype, np.floating) and not np.isfinite(arr).all():
            return "non-finite pixel values (NaN or Inf)"
        return None

    @staticmethod
    def check_flat(arr: np.ndarray, min_std: float) -> Optional[str]:
        """Reject solid or near-solid images.

        Args:
            arr: uint8 pixel array of shape (H, W, 3).
            min_std: Minimum standard deviation of pixel values.

        Returns:
            A reason string if the image is flat, otherwise None.
        """
        std = float(arr.std())
        if std < min_std:
            return f"near-flat image (std {std:.1f})"
        return None

    @staticmethod
    def check_robust_range(arr: np.ndarray, min_range: float) -> Optional[str]:
        """Reject images with almost no spread between dark and light pixels, ignoring stray extreme pixels.

        Args:
            arr: uint8 pixel array of shape (H, W, 3).
            min_range: Minimum spread between the 1st and 99th percentile pixel values.

        Returns:
            A reason string if the spread is too small, otherwise None.
        """
        low, high = np.percentile(arr, [1, 99])
        if high - low < min_range:
            return f"low robust range ({high - low:.1f} between 1st and 99th percentile)"
        return None

    @staticmethod
    def check_dominant_colour(arr: np.ndarray, max_fraction: float) -> Optional[str]:
        """Reject images where almost every pixel is the same colour.

        Colours are quantised to 16 levels per channel before counting.

        Args:
            arr: uint8 pixel array of shape (H, W, 3).
            max_fraction: Largest allowed share of pixels in the most common colour.

        Returns:
            A reason string if one colour dominates, otherwise None.
        """
        quantised = (arr >> 4).astype(np.uint16)
        codes = (quantised[..., 0] << 8) | (quantised[..., 1] << 4) | quantised[..., 2]
        dominant = float(np.bincount(codes.ravel()).max() / codes.size)
        if dominant > max_fraction:
            return f"{dominant:.1%} of pixels are one colour"
        return None

    # ------------------------------------------------------------------ orchestration

    def check_image(self, reference: ImageReference) -> ImageReport:
        """Open one image once and run every check on it, collecting issues along the way.

        Nothing is moved or written. An undecodable file gets a single issue and no further checks.

        Args:
            reference: The image to check.

        Returns:
            A report with the issues, the warnings, and the pixel hash if the image passed everything.
        """
        path: Path = reference.resolved_filepath
        cfg = self.config

        try:
            loaded = self.load_image(path)
        except Exception as error:  # any decode failure means the file is unusable for training
            issue = f"could not be decoded ({type(error).__name__}: {error})"
            return ImageReport(reference, (issue,), (), None, None)

        thumbnail = self.to_thumbnail_array(loaded.pixels, cfg.thumbnail_size)
        outcomes: list[Optional[str]] = [
            self.check_min_size(loaded.pixels, cfg.img_size),
            self.check_aspect_ratio(loaded.pixels, cfg.max_aspect_ratio),
            self.check_animated(loaded),
            self.check_transparency(loaded, cfg.max_transparent_fraction),
            self.check_finite(thumbnail),
            self.check_flat(thumbnail, cfg.min_std),
            self.check_robust_range(thumbnail, cfg.min_robust_range),
            self.check_dominant_colour(thumbnail, cfg.max_dominant_fraction),
        ]
        issues = tuple(outcome for outcome in outcomes if outcome is not None)

        mismatch = self.check_format_matches_extension(loaded, path)
        warning_list = (mismatch,) if mismatch is not None else ()

        digest = None if issues else self.pixel_sha256(loaded.pixels)
        return ImageReport(reference, issues, warning_list, digest, loaded.exif_orientation)











    #
    # # def _find_non_approved_files(self) -> QuarantineReport:
    # #     """
    # #     Run this after the directory scanner tool.
    # #
    # #     Assumumption is that when running this method, the dir layout is ok.
    # #     We're now going to inspect the file contents and tell non approved typed image files they gotsta go!
    # #     """
    # #     quarantine_entry_list: list[QuarantineEntry] = []
    # #     # Skipping system files, show me all files in the approved directory tree.
    # #     for file in self._iter_raw_files():
    # #         label, source, path = file
    # #         # TODO: what if suffix doesn't exist? that would raise attribute error, no?
    # #         #   if its not a system file we're avoiding, quaranting it!
    # #         if path.suffix.lower() not in AcceptedImageFormats.as_tuple(include_dot=True):
    # #             quarantine = QuarantineEntry(
    # #                 reference=ImageDatasetFileReference(
    # #                     images_raw_dir=self.images_raw_dir,
    # #                     source=source,
    # #                     label=label,
    # #                     filename=path.name,
    # #                 ),
    # #                 reasons=(f"{label.value}/{source.value}/{path.name} is not of an acceptable dataset image type", ),
    # #                 duplicate_group=None
    # #             )
    # #             quarantine_list.append(quarantine)
    # #
    # #     return QuarantineReport(
    # #         stage=QuarantineStage.FILESYSTEM__DIR_STRUCTURE_CHECK,
    # #         entries=tuple(quarantine_list)
    # #     )
    #
    #
    # # So now at this point in the pipeline, we've estabilished:
    # #   - the desired dir structure exists
    # #   - no superfluous dirs or sub directories exist
    # #   - we've identified non approved image files
    # #   - if we quarantined all the non approved image files, and ignore system files,
    # #     we're left with the actual dataset (as naive image files, yet to be evaluated)
    # #   -
    #
    #
    # # def scan_dataset_get_quarantine_report(self) -> QuarantineReport:
    # #     # 1. Ensure desired directory structure exists
    # #     self._create_image_raw_subdirectories()
    # #
    # #     # 2. Ensure the desired directory structure ONLY exists.
    # #     #    Missing or superfluous directories will cause RAISE OUT here
    # #     self._assert_images_raw_layout_is_valid()
    # #
    # #     # 3. Now that we have the desired dataset directory structure,
    # #     #    Let's find any non-system files that aren't image types we want
    # #     #    and create a quarantine report for them
    # #     non_approved_image_files: QuarantineReport = self._find_non_approved_files()
    # #
    # #     logger.info(
    # #         f"Found {len(non_approved_image_files.entries)} files to quarantine from the dataset directory"
    # #     )
    # #     return non_approved_image_files
    # #
    # #
    # # def check_file_extension_lowercases(self) -> None:
    # #     for file in self._iter_raw_files():
    # #         label, source, path = file
    # #         if not path.suffix:
    # #             logger.warning(f"File {path} does not have an extension, skipping")
    # #             continue
    # #
    # #         if path.suffix.lower() not in AcceptedImageFormats.as_tuple(include_dot=True):
    # #             logger.warning(f"File {path} is not of image file of acceptable format, skipping")
    # #             continue
    # #
    # #         if path.suffix != path.suffix.lower():
    # #             logger.warning(f"File {path} is an image file but the extension is not fully lowercase.")
    # #             # TODO: check if dry run
    # #             #   move to a repair manager?
    # #             #   if has_renamed_extension; scan the dir again!
    # #             rename_file_with_lowercase_extension(
    # #                 file_path=path,
    # #                 dry_run=False
    # #             )
    # #             # TODO: you'd need to re-check these renamed files, too. grrrrr!!
    # #             # ideally before the heavier image checks
    #
    #
    # def gen_list_image_refs(self) -> list[ImageDatasetFileReference]:
    #     """
    #     RUN THIS METHOD AFTER clean_dataset()!!!
    #
    #     THIS METHOD ALLOWS YOU TO GET A LIST OF ImageReference OBJECTS THAT MAP YOUR DATASET!
    #     Create this list of ImageReferences and pass that into the DatasetImageQualityManager class.
    #
    #     with a clean raw images dataset directory, we can now generate our reference objects
    #
    #     this output will then feed into the existing DataQualityManager class
    #     which performs checks on the actual image file and contents.
    #     We don't need to worry about that implementation here.
    #
    #     Returns:
    #         One ImageReference per file under images_raw_dir/<label>/<source>/.
    #     """
    #     references: list[ImageDatasetFileReference] = [
    #         ImageDatasetFileReference(
    #             images_raw_dir=self.images_raw_dir,
    #             source=source,
    #             label=label,
    #             filename=path.name
    #         )
    #         for label, source, path in self._iter_raw_files()
    #     ]
    #     logger.info(f"Generated {len(references)} image references from {self.images_raw_dir}")
    #     return references
