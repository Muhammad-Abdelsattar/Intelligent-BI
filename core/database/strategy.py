from sqlalchemy import create_engine, inspect

# --- Abstract Strategy ---
class DatabaseConnectionStrategy(ABC):
    """Abstract base class for a database connection strategy."""
    @abstractmethod
    def get_uri(self) -> str:
        """Constructs the SQLAlchemy database URI."""
        pass

    @abstractmethod
    def get_schema_info(self) -> str:
        """Fetches and returns the database schema information as a string."""
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

    def get_schema_info(self) -> str:
        engine = create_engine(self.get_uri())
        inspector = inspect(engine)
        schema_info = []
        for table_name in inspector.get_table_names():
            schema_info.append(f"TABLE {table_name}:")
            for column in inspector.get_columns(table_name):
                schema_info.append(f"  {column['name']} {column['type']}")
        return "\n".join(schema_info)

@dataclass
class SqliteConnectionStrategy(DatabaseConnectionStrategy):
    """Strategy for connecting to a SQLite database."""
    db_path: str

    def get_uri(self) -> str:
        return f"sqlite:///{self.db_path}"

    def get_schema_info(self) -> str:
        engine = create_engine(self.get_uri())
        inspector = inspect(engine)
        schema_info = []
        for table_name in inspector.get_table_names():
            schema_info.append(f"TABLE {table_name}:")
            for column in inspector.get_columns(table_name):
                schema_info.append(f"  {column['name']} {column['type']}")
        return "\n".join(schema_info)
