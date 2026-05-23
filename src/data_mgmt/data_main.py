from config.globals import SourceLabels, ClassLabels
from config.paths import LOGS_DIR, DB_PATH
from data_mgmt.data_reconciler import Reconciler, ScanTarget, build_scan_targets
from data_mgmt.image_database import DatabaseManager
from monitoring.logging_manager import LoggingManager


def prepare_training_data_db_pipeline():
    """
    Ingest locally stored training data into the training data image metadata database.
    """
    # Init logging manager so that the process is recorded
    LoggingManager(
        log_dir=LOGS_DIR,
        session_name="prepare_new_training_db_table"
    )

    # Init the database CRUD manager
    db_manager = DatabaseManager(db_path=DB_PATH)

    # Build scan targets
    scan_targets: list[ScanTarget] = build_scan_targets(
        labels=[ClassLabels.ACOUSTIC, ClassLabels.ELECTRIC, ClassLabels.NOT_GUITAR],
        sources=[SourceLabels.UNSPLASH, SourceLabels.PIXABAY, SourceLabels.MIXED],
    )

    # Init reconciler
    reconciler = Reconciler(
        db_manager=db_manager,
        dry_run=False
    )

    # Run the ingestion and reconciliation logic
    reconciler.run(targets=scan_targets)


if __name__ == "__main__":
    prepare_training_data_db_pipeline()
