import pandas as pd
from omegaconf import DictConfig
import sqlparse

from .manager import DatabaseManager
from .validator import SQLValidator
from .executor import DatabaseExecutor

class DatabaseService:
    """
    A high-level service that orchestrates database operations,
    combining connection management, query validation, and execution.
    """
    def __init__(self, db_config: DictConfig):
        """
        Initializes the DatabaseService by setting up the manager, validator, and executor.

        Args:
            db_config: OmegaConf DictConfig containing database connection details.
        """
        self._manager = DatabaseManager.from_config(db_config)
        self._validator = SQLValidator(self._manager.get_uri())
        self._executor = DatabaseExecutor(self._manager.get_engine())

    def execute_validated_query(self, sql_query: str) -> pd.DataFrame:
        """
        Validates an SQL query and, if valid, executes it.

        Args:
            sql_query: The SQL query string to validate and execute.

        Returns:
            A pandas DataFrame containing the query results.

        Raises:
            ValueError: If the query is invalid or not a SELECT statement.
            RuntimeError: If there is a database execution error.
        """
        is_valid, message = self._validator.verify_executable_select_query(sql_query)
        if not is_valid:
            raise ValueError(f"Invalid SQL query: {message}")

        return self._executor.execute_query(sql_query)

    def get_schema_info(self) -> str:
        """
        Retrieves the database schema information from the underlying strategy.

        Returns:
            A string representation of the database schema.
        """
        return self._manager.get_schema_info()
