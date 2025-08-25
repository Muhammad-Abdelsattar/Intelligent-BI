from omegaconf import DictConfig
import pandas as pd

from core.llm import LLMService
from core.database import DatabaseService

class SQLGeneratorAgent:
    """
    An agent responsible for generating and executing SQL queries based on natural language questions.
    """
    def __init__(self, agent_config: DictConfig, db_config: DictConfig):
        """
        Initializes the SQLGeneratorAgent.

        Args:
            agent_config: Configuration specific to this agent (e.g., LLM provider key).
            db_config: Configuration for the database connection.
        """
        self.llm_service = LLMService(
            agent_name=agent_config.name, # Now configurable
            provider_key=agent_config.llm_provider_key,
            llm_config_path=agent_config.llm_config_path,
            prompts_base_path=agent_config.prompts_base_path
        )

        self.db_service = DatabaseService(db_config)

        self.agent_config = agent_config

    def generate_and_execute_sql(self, natural_language_question: str, database_dialect: str) -> pd.DataFrame:
        """
        Generates an SQL query from a natural language question, validates it, and executes it.

        Args:
            natural_language_question: The question in natural language.
            database_dialect: The SQL dialect of the target database (e.g., "PostgreSQL", "SQLite").

        Returns:
            A pandas DataFrame containing the results of the SQL query.

        Raises:
            ValueError: If the LLM fails to generate SQL or validation fails.
            RuntimeError: If SQL execution fails.
        """
        # 1. Get database schema
        schema_definition = self.db_service.get_schema_info()

        # 2. Prepare variables for LLM prompt
        llm_variables = {
            "schema_definition": schema_definition,
            "user_question": natural_language_question,
            "database_dialect": database_dialect
        }

        # 3. Generate SQL query using LLMService
        generated_sql = self.llm_service.generate_text(llm_variables)

        # Check if LLM returned an error message (as per prompt instructions)
        if generated_sql.strip().startswith("Error:"):
            raise ValueError(f"LLM failed to generate SQL: {generated_sql}")

        # 4. Validate and execute SQL query using DatabaseService
        try:
            results_df = self.db_service.execute_validated_query(generated_sql)
            return results_df
        except ValueError as e:
            raise ValueError(f"Generated SQL validation failed: {e}. Query: {generated_sql}")
        except RuntimeError as e:
            raise RuntimeError(f"SQL execution failed: {e}. Query: {generated_sql}")
