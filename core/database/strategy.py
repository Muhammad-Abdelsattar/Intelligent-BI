from abc import ABC, abstractmethod
from dataclasses import dataclass

# --- Abstract Strategy ---
class DatabaseConnectionStrategy(ABC):
    """Abstract base class for a database connection strategy."""
    @abstractmethod
    def get_uri(self) -> str:
        """Constructs the SQLAlchemy database URI."""
        pass

# --- Concrete Strategies ---
@dataclass
class PostgresConnectionStrategy(DatabaseConnectionStrategy):
    """Strategy for connecting to a PostgreSQL database."""
    host: str
    port: int
    user: str
    password: str
    dbname: str

    def get_uri(self) -> str:
        # Assumes psycopg2 driver
        return f"postgresql+psycopg2://{self.user}:{self.password}@{self.host}:{self.port}/{self.dbname}"

@dataclass
class SqliteConnectionStrategy(DatabaseConnectionStrategy):
    """Strategy for connecting to a SQLite database."""
    db_path: str

    def get_uri(self) -> str:
        return f"sqlite:///{self.db_path}"