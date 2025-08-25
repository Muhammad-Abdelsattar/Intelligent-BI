import pandas as pd
from sqlalchemy.engine import Engine
from .manager import DatabaseManager

class DatabaseExecutor:
    """
    Handles the execution of SQL queries against the database.
    """
    def __init__(self, engine: Engine):
        self._engine = engine

    def execute_query(self, sql_query: str) -> pd.DataFrame:
        """
        Executes a read-only SQL query and returns the results as a pandas DataFrame.

        Args:
            sql_query: The SQL SELECT statement to execute.

        Returns:
            A pandas DataFrame containing the query results.

        Raises:
            RuntimeError: If there is a database execution error.
        """
        try:
            with self._engine.connect() as connection:
                df = pd.read_sql_query(sql_query, connection)
            return df
        except SQLAlchemyError as e:
            raise RuntimeError(f"Database execution failed: {e}") from e
