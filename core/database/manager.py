from __future__ import annotations
from omegaconf import DictConfig
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine

from .strategy import DatabaseConnectionStrategy, PostgresConnectionStrategy, SqliteConnectionStrategy

class DatabaseManager:
    """
    Manages database connections via a strategy.
    """
    def __init__(self, db_config: DictConfig):
        """
        Initializes the DatabaseManager with the correct strategy
        based on the provided configuration.
        """
        db_type = db_config.get("type")
        if not db_type:
            raise ValueError("Database 'type' must be specified in the configuration.")

        strategy: DatabaseConnectionStrategy
        params = db_config.get("params", {})

        if db_type == "postgres":
            strategy = PostgresConnectionStrategy(**params)
        elif db_type == "sqlite":
            strategy = SqliteConnectionStrategy(**params)
        else:
            raise NotImplementedError(f"Database type '{db_type}' is not supported.")

        self._strategy = strategy
        self._engine = create_engine(self._strategy.get_uri())

    def get_engine(self) -> Engine:
        """Returns the SQLAlchemy engine instance."""
        return self._engine

    def get_uri(self) -> str:
        """Returns the database connection URI."""
        return self._strategy.get_uri()

    def get_schema_info(self) -> str:
        """Returns the database schema information from the strategy."""
        return self._strategy.get_schema_info()
