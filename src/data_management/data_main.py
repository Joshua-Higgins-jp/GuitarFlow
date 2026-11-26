from pathlib import Path
from typing import Optional

from loguru import logger

from common.constants.session_id import SessionID
from data_management.dataset_file_reference import DatasetFileReference
from common.constants.job_names import JobNames
from common.dirs_paths import DirsPaths
from common.logging_add_sink import add_log_sink
from common.logging_set_log_filename import gen_log_filepath
from common.utils.dt_get import get_dt_now_jst
from data_management.validate_dataset_dir_manager import DatasetValidationDirFilesManager, ValidateDirStructureResult
from data_management.quarantine import QuarantineReport


# from data_management.db_crud import DatabaseCRUD
# from data_management.db_manager import DatabaseManager


def pipeline_01__image_dataset_validation(dry_run: bool = True) -> None:
    """

    """
    job_name = JobNames.CLEAN_DATASET

    # 1. Create Session ID object
    session_id: SessionID = SessionID(
        session_start_time=get_dt_now_jst(),
        job_name=job_name
    )

    # 2. Init logging
    log_filepath: Path = gen_log_filepath(session_id=session_id)
    add_log_sink(log_filepath=log_filepath)

    logger.info(f"Session ID {session_id.session_id_str} starting job {job_name} with log file {log_filepath}")

    # 3. Init and validate dirs and paths
    dirs_paths: DirsPaths = DirsPaths.build()
    dirs_paths.log_paths()

    ########## ========================= ##########

    # 4. Check local file system's image dataset
    file_manager: DatasetValidationDirFilesManager = DatasetValidationDirFilesManager(images_raw_dir=dirs_paths.IMG_RAW_DIR)
    file_manager_s1_validate_dirs: ValidateDirStructureResult = file_manager.validate_dir_structure()
    if file_manager_s1_validate_dirs.has_problems:
        raise RuntimeError(file_manager_s1_validate_dirs.runtime_error_str())

    # file_manager.check_file_extension_lowercases()

    # surviving_image_refs: list[ImageReference] = file_manager.gen_list_image_refs()

    # TODO: you need to perform image content quality checks, e.g. slender images with bad aspect ratio, low color variation, etc.
    #   images with matching hash both get kicked to quarantine. You could implement similar images algorithm but may be overkill for mvp.
    #   you'd log data quality quarantines in a manifesto file and also move those images to quarantine.

    # TODO: Then you init the database and crud manager

    # TODO: then you'd insert the surviging image refs from data quality pass into the database.
    # At which point, you're ready to move on to model training!
    #   prepare dataset splits, collect metrics,

    # TODO: current issue - deciding how to implement logic between classes, or all in 1 class,

    # # 4. Init the database manager, which validates database
    # db_manager = DatabaseManager.init_database(db_path=dirs_paths.DB_PATH)
    #
    # # 5. Init the crud manager
    # db_crud = DatabaseCRUD(db_manager=db_manager)



    # # Build scan targets
    # scan_targets: list[ScanTarget] = build_scan_targets(
    #     labels=[ClassLabels.ACOUSTIC, ClassLabels.ELECTRIC, ClassLabels.NOT_GUITAR],
    #     sources=[SourceLabels.UNSPLASH, SourceLabels.PIXABAY, SourceLabels.MIXED],
    # )
    #
    # # Init reconciler
    # reconciler = Reconciler(
    #     db_manager=db_manager,
    #     dry_run=False
    # )
    #
    # # Run the ingestion and reconciliation logic
    # reconciler.run(targets=scan_targets)


if __name__ == "__main__":
    pipeline_01__image_dataset_validation()
