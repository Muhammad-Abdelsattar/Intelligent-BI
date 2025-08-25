from __future__ import annotations
from omegaconf import DictConfig
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine

from .strategy import DatabaseConnectionStrategy, PostgresConnectionStrategy, SqliteConnectionStrategy

class DatabaseManager:
    """
    Manages database connections via a strategy and acts as a factory for itself.
    """
    def __init__(self, strategy: DatabaseConnectionStrategy):
        if not isinstance(strategy, DatabaseConnectionStrategy):
            raise TypeError("A valid DatabaseConnectionStrategy object must be provided.")
        self._strategy = strategy
        self._engine = create_engine(self._strategy.get_uri())

    @classmethod
    def from_config(cls, db_config: DictConfig) -> DatabaseManager:
        """
        Factory class method to build a DatabaseManager with the correct strategy
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

        return cls(strategy)

    def get_engine(self) -> Engine:
        """Returns the SQLAlchemy engine instance."""
        return self._engine

    def get_uri(self) -> str:
        """Returns the database connection URI."""
        return self._strategy.get_uri()

    def get_schema_info(self) -> str:
        """Returns the database schema information from the strategy."""
        return self._strategy.get_schema_info()
