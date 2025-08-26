import pandas as pd
import sqlparse
from omegaconf import DictConfig
from typing import Any, Dict, Optional, List
from sqlalchemy.exc import SQLAlchemyError
from langchain_community.utilities.sql_database import SQLDatabase

from .manager import DatabaseManager

class AgentDatabaseService:
    """
    A self-contained, unified database service that provides interfaces for
    both AI agents (string-based) and internal programmatic use (DataFrame-based).
    """
    def __init__(
        self,
        db_config: DictConfig,
        include_tables: Optional[List[str]] = None,
        sample_rows_in_table_info: int = 3
    ):
        """
        Initializes the service.

        Args:
            db_config: OmegaConf DictConfig containing database connection details.
            include_tables: Optional list of table names to expose to the agent.
            sample_rows_in_table_info: Number of sample rows to include in schema info.
        """
        self.db_type = db_config.get("type"," ")
        self._manager = DatabaseManager(db_config)
        self._engine = self._manager.get_engine()

        self._langchain_db = SQLDatabase(
            engine=self._engine,
            sample_rows_in_table_info=sample_rows_in_table_info,
            include_tables=include_tables,
        )
        self.dialect = self._langchain_db.dialect


    def get_context_for_agent(self) -> Dict[str, Any]:
        """Gets all necessary context (schema, tables) for a prompt template."""
        context = {**self._langchain_db.get_context(),"database_dialect": self.dialect }
        return context

    def run_query_for_agent(self, sql_query: str) -> str:
        """
        Executes a query for an agent. Returns results or errors as a string.
        This method is safe and will not crash an agent's tool run.
        """
        return self._langchain_db.run_no_throw(sql_query)


    def execute_for_dataframe(self, sql_query: str) -> pd.DataFrame:
        """
        Executes a read-only SQL query and returns the results as a pandas DataFrame.
        This method contains its own execution logic and is for internal application use.

        Args:
            sql_query: The SQL query string to execute.

        Returns:
            A pandas DataFrame containing the query results.

        Raises:
            ValueError: If the query is not a SELECT statement.
            RuntimeError: If there is a database execution error.
        """
        # A simple security check to prevent modifications
        try:
            statement_type = sqlparse.parse(sql_query)[0].get_type()
            if statement_type != 'SELECT':
                raise ValueError(
                    f"Only SELECT statements are allowed for DataFrame execution. "
                    f"Found statement of type: {statement_type}"
                )
        except IndexError:
            raise ValueError("The SQL query is empty or invalid.")

        # Execute the query and return a DataFrame
        try:
            with self._engine.connect() as connection:
                df = pd.read_sql_query(sql_query, connection)
            return df
        except SQLAlchemyError as e:
            raise RuntimeError(f"Database execution failed: {e}") from e