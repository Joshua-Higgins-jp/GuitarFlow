from enum import StrEnum


class JobNames(StrEnum):
    CLEAN_DATASET = "clean_dataset"
    INGEST_DATASET_TO_DB = "ingest_dataset_to_db"
