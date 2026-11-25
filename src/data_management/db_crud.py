from data_management.db_manager import DatabaseManager


class DatabaseCRUD:
    """
    Database CRUD operations logic, which builds on the connection made in the manager.
    """
    def __init__(
            self,
            db_manager: DatabaseManager
    ) -> None:
        """
        Store the database handle without touching the database.

        Args:
            db_manager: A DatabaseManager whose schema has already been verified.
        """
        self._db: DatabaseManager = db_manager
