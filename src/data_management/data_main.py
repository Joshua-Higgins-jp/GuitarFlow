from pathlib import Path

from loguru import logger

from common.constants.job_names import JobNames
from common.constants.session_id import SessionID
from common.dirs_paths import DirsPaths
from common.logging_add_sink import add_log_sink
from common.logging_set_log_filename import gen_log_filepath
from common.utils.dt_get import get_dt_now_jst
from data_management.quarantine_manager import QuarantineManager
from data_management.validate_dataset_dir_manager import (
    DatasetValidationDirFilesManager,
    TargetsAndNonTargetsSets
)
from data_management.validate_image_manager import (
    DatasetValidationImageCheckManager,
    ImageCheckConfig,
    ImageChecksResultsSets
)


def pipeline_01__clean_dataset(dry_run: bool = True) -> None:
    """
    docstring
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

    logger.info(f"Session ID {session_id.session_id_base} starting job {job_name} with log file {log_filepath}")

    # 3. Init and validate dirs and paths
    dirs_paths: DirsPaths = DirsPaths.build()
    dirs_paths.log_paths()

    ########## ========================= ##########

    # 4. Check local file system's image dataset
    file_manager: DatasetValidationDirFilesManager = DatasetValidationDirFilesManager(images_raw_dir=dirs_paths.IMG_RAW_DIR)
    target_sets: TargetsAndNonTargetsSets = file_manager.gen_target_sets()
    target_sets.log_results()

    image_validation = DatasetValidationImageCheckManager(
        dataset_targets_sets=target_sets,
        config=ImageCheckConfig(),
    )
    image_check_sets: ImageChecksResultsSets = image_validation.run_checks()

    quarantine_manager: QuarantineManager = QuarantineManager(
        failed_image_checks=image_check_sets.failed,
        quarantine_dir=dirs_paths.QUARANTINE_DIR
    )
    quarantine_report_csv: Path = quarantine_manager.run_quarantine(quarantine_csv_filename=session_id.session_id_quarantine)

    # # 4. Init the database manager, which validates database
    # db_manager = DatabaseManager.init_database(db_path=dirs_paths.DB_PATH)
    #
    # # 5. Init the crud manager
    # db_crud = DatabaseCRUD(db_manager=db_manager)

    # TODO: then you'd insert the surviging image refs from data quality pass into the database.
    # At which point, you're ready to move on to model training!
    #   prepare dataset splits, collect metrics,

    # TODO: current issue - deciding how to implement logic between classes, or all in 1 class,


if __name__ == "__main__":
    pipeline_01__clean_dataset()
