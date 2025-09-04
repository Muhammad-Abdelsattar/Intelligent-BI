from __future__ import annotations
import logging
from omegaconf import DictConfig
import pandas as pd
from pathlib import Path
from typing import List, TypedDict, Optional, Dict, Any, Tuple
from langgraph.graph import StateGraph, END

from core.llm import LLMService
from core.database import AgentDatabaseService
from core.models.sql_agent_models import SQLAgentResponse


class SQLAgentState(TypedDict):
    """
    Represents the internal, temporary state of the SQL Agent's workflow.
    This state is created at the start of a `run` and destroyed at the end.
    """
    # Inputs for the workflow
    natural_language_question: str
    chat_history: List[Tuple[str, str]] # The history of the conversation

    # Database context
    db_context: Dict[str, Any]

    # Internal loop state
    history: List[str] # The history of generation attempts and errors in this run
    max_attempts: int
    current_attempt: int

    # Outputs
    generated_sql: str
    final_dataframe: Optional[pd.DataFrame]
    error: Optional[str]


logger = logging.getLogger(__name__)


class SQLAgent:
    """
    A self-contained agent that generates and executes SQL queries.

    This agent is designed to be called by a higher-level orchestrator. It accepts
    a user question and optional conversational history, and manages an internal
    retry loop to robustly generate a valid SQL query. Its final output is a
    pandas DataFrame.
    """

    def __init__(
        self,
        llm_service: LLMService,
        db_service: AgentDatabaseService,
        name: str,
        max_attempts: int,
    ):
        self.llm_service = llm_service
        self.db_service = db_service
        self.name = name
        self.max_attempts = max_attempts
        self.workflow = self._build_graph()
        self.app = self.workflow.compile()


    @classmethod
    def from_config(
        cls,
        agent_key: str,
        app_config: DictConfig,
        prompts_base_path: Path,
        db_key_override: Optional[str] = None
    ) -> "SQLAgent":
        agent_config = app_config.agents.get(agent_key)
        if not agent_config:
            raise ValueError(f"Agent key '{agent_key}' not found in agents configuration.")
        
        llm_service = LLMService(
            agent_prompts_dir=agent_config.prompts_dir,
            provider_key=agent_config.llm_provider_key,
            llm_config=app_config.llms,
            prompts_base_path=prompts_base_path
        )
        
        # Use the override if provided, otherwise fall back to the agent's default
        db_key = db_key_override if db_key_override is not None else agent_config.database_key
        
        db_config = app_config.databases.get(db_key)
        if db_config is None:
            raise ValueError(f"Database key '{db_key}' not found in databases configuration.")
            
        db_service = AgentDatabaseService(db_config)
        
        return cls(
            llm_service=llm_service,
            db_service=db_service,
            name=agent_config.name,
            max_attempts=agent_config.max_attempts
        )

    def _build_graph(self) -> StateGraph:
        # The graph structure remains unchanged.
        graph = StateGraph(SQLAgentState)
        graph.add_node("generate_sql", self.generate_sql_node)
        graph.add_node("execute_sql", self.execute_sql_node)
        graph.set_entry_point("generate_sql")
        graph.add_edge("generate_sql", "execute_sql")
        graph.add_conditional_edges(
            "execute_sql",
            self.should_retry_node,
            {"retry": "generate_sql", "end": END}
        )
        return graph

    def generate_sql_node(self, state: SQLAgentState) -> Dict[str, Any]:
        """
        Node that generates an SQL query, now including conversational history.
        """
        logger.info(f"--- Attempt {state['current_attempt'] + 1} of {state['max_attempts']}: Generating SQL ---")

        # Format the conversational history into a string for the prompt
        chat_history_str = "\n".join(
            [f"Human: {q}\nAI: {a}" for q, a in state.get("chat_history", [])]
        ).strip()

        llm_variables = {
            "database_dialect": self.db_service.dialect,
            "schema_definition": state["db_context"]["table_info"],
            "user_question": state["natural_language_question"],
            "history": "\n".join(state["history"]), # Internal retry history
            "chat_history": chat_history_str if chat_history_str else "No previous conversation history."
        }


        response: SQLAgentResponse = self.llm_service.generate_structured(
            variables=llm_variables,
            response_model=SQLAgentResponse
        )

        if response.status == 'success':
            generated_sql = response.query
            logger.info(f"Successfully generated SQL:\n{generated_sql}")
            history = state["history"] + [f"ATTEMPT {state['current_attempt'] + 1} SQL:\n{generated_sql}"]
            return {
                "current_attempt": state["current_attempt"] + 1,
                "generated_sql": generated_sql,
                "history": history
            }
        else:
            error_message = f"Error: {response.reason}"
            logger.warning(f"LLM refused to generate SQL, returning structured error: {error_message}")
            history = state["history"] + [f"ATTEMPT {state['current_attempt'] + 1} LLM ERROR:\n{error_message}"]
            return {
                "current_attempt": state["current_attempt"] + 1,
                "generated_sql": error_message,
                "history": history
            }

    def execute_sql_node(self, state: SQLAgentState) -> Dict[str, Any]:
        # This node remains unchanged.
        generated_sql = state["generated_sql"].strip()
        if generated_sql.startswith("Error:"):
            logger.warning(f"LLM refused to generate SQL, returning error: {generated_sql}")
            return {"final_dataframe": None, "error": generated_sql}
        logger.info("--- Executing SQL ---")
        try:
            results_df = self.db_service.execute_for_dataframe(generated_sql)
            logger.info("Successfully executed SQL.")
            return {"final_dataframe": results_df, "error": None}
        except (ValueError, RuntimeError) as e:
            error_message = f"Error executing SQL: {e}"
            logger.error(f"Execution failed with error:\n{error_message}")
            history = state["history"] + [f"DATABASE ERROR:\n{error_message}"]
            return {"error": error_message, "history": history, "final_dataframe": None}

    def should_retry_node(self, state: SQLAgentState) -> str:
        # This node remains unchanged.
        error = state.get("error")
        if error is None:
            logger.info("--- Workflow successful ---")
            return "end"
        if error.strip() == state.get("generated_sql").strip():
            logger.error(f"Workflow ended because LLM returned a deliberate error: {error}")
            return "end"
        if state["current_attempt"] >= state["max_attempts"]:
            logger.warning("-- Max attempts reached, ending workflow --")
            return "end"
        logger.info(f"--- Database error found, retrying ---")
        return "retry"

    def run(self, natural_language_question: str, chat_history: Optional[List[Tuple[str, str]]] = None) -> pd.DataFrame:
        """
        Executes the agent's workflow to answer a natural language question,
        considering the provided conversational history.

        Args:
            natural_language_question: The user's current question.
            chat_history: An optional list of (human, ai) tuples representing
                          the conversation so far.

        Returns:
            A pandas DataFrame with the query results.

        Raises:
            RuntimeError: If the agent fails to produce a valid SQL query
                          after all attempts.
        """
        # The agent receives the chat_history and uses it to create the initial state for the graph.
        initial_state: SQLAgentState = {
            "natural_language_question": natural_language_question,
            "chat_history": chat_history or [],
            "db_context": self.db_service.get_context_for_agent(),
            "history": [],
            "max_attempts": self.max_attempts,
            "current_attempt": 0,
            "generated_sql": "",
            "final_dataframe": None,
            "error": None
        }

        # The internal graph runs with this temporary, complete state.
        final_state = self.app.invoke(initial_state)

        # The agent returns its primary output, as expected by the orchestrator.
        if final_state["final_dataframe"] is not None:
            return {"dataframe":final_state["final_dataframe"],"sql_query":final_state["generated_sql"]}
        else:
            final_error = final_state["error"] or "Unknown error"
            raise RuntimeError(
                # f"Agent failed to generate a valid SQL query after {self.max_attempts} attempts. "
                f"Agent failed to generate a valid SQL query after the maximum number of attempts. "
                f"Final error: {final_error}"
            )
