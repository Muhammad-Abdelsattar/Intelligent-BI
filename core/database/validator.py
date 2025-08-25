from sqlalchemy import create_engine, text
from sqlalchemy.exc import ProgrammingError

class SQLValidator:
    """
    Validates SQL queries by attempting to generate an execution plan using EXPLAIN.
    """
    def __init__(self, db_uri: str):
        """
        Initializes the validator with a database connection URI.

        Args:
            db_uri: A SQLAlchemy-compatible database URI
                    (e.g., 'sqlite:///mydatabase.db', 'postgresql://user:pass@host/dbname').
        """
        try:
            self.engine = create_engine(db_uri)
        except ImportError:
            raise ImportError("SQLAlchemy is not installed. Please install it with 'pip install SQLAlchemy'.")

    def verify_executable_select_query(self, sql_query: str) -> tuple[bool, str]:
        """
        Validates the SQL query syntax by asking the database to create an
        execution plan. It only validates SELECT statements.

        Args:
            sql_query: The SQL query string to validate.

        Returns:
            A tuple (is_valid: bool, message: str).
            The message will be 'OK' on success or an error message on failure.
        """
        try:
            statement_type = sqlparse.parse(sql_query)[0].get_type()
            if statement_type != 'SELECT':
                return False, f"Error: Query must be a SELECT statement, but it is a {statement_type} statement."
        except IndexError:
            return False, "Error: The SQL query is empty or invalid."

        query = sql_query.strip()

        try:
            with self.engine.connect() as connection:
                # Using text() is important for safely passing the query.
                # The EXPLAIN statement asks the DB to parse and plan the query.
                connection.execute(text(f"EXPLAIN {query}"))
            return True, "OK"
        except ProgrammingError as e:
            # This exception is commonly raised for syntax errors.
            return False, f"Syntax Error: {e.orig}"
        except Exception as e:
            return False, f"An unexpected error occurred during validation: {e}"
