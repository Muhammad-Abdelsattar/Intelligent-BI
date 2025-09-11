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

    # Inputs
    natural_language_question: str
    chat_history: List[Tuple[str, str]]
    db_context: Dict[str, Any]

    # Internal loop state
    history: List[str]
    max_attempts: int
    current_attempt: int

    # The structured result from the generation LLM
    generation_result: Optional[SQLAgentResponse]

    # Final Outputs
    final_dataframe: Optional[pd.DataFrame]
    generated_sql: str
    error: Optional[str]


logger = logging.getLogger(__name__)


class SQLAgent:
    """
    A self-contained agent that generates and executes SQL queries.

    This agent manages an internal retry loop and can gracefully handle ambiguity
    by requesting clarification. Its final output is a structured dictionary
    indicating success, a need for clarification, or an error.
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
        db_key_override: Optional[str] = None,
    ) -> SQLAgent:
        # ... (This method remains correct and does not need changes) ...
        agent_config = app_config.agents.get(agent_key)
        if not agent_config:
            raise ValueError(
                f"Agent key '{agent_key}' not found in agents configuration."
            )

        llm_service = LLMService(
            agent_prompts_dir=agent_config.prompts_dir,
            provider_key=agent_config.llm_provider_key,
            llm_config=app_config.llms,
            prompts_base_path=prompts_base_path,
        )
        db_key = (
            db_key_override
            if db_key_override is not None
            else agent_config.database_key
        )
        db_config = app_config.databases.get(db_key)
        if db_config is None:
            raise ValueError(
                f"Database key '{db_key}' not found in databases configuration."
            )
        db_service = AgentDatabaseService(db_config)
        return cls(
            llm_service=llm_service,
            db_service=db_service,
            name=agent_config.name,
            max_attempts=agent_config.max_attempts,
        )

    def _build_graph(self) -> StateGraph:
        graph = StateGraph(SQLAgentState)
        graph.add_node("generate_sql", self.generate_sql_node)
        graph.add_node("execute_sql", self.execute_sql_node)
        graph.set_entry_point("generate_sql")
        graph.add_edge("generate_sql", "execute_sql")
        graph.add_conditional_edges(
            "execute_sql", self.should_retry_node, {"retry": "generate_sql", "end": END}
        )
        return graph

    def generate_sql_node(self, state: SQLAgentState) -> Dict[str, Any]:
        """Node that generates an SQL query and saves the full structured response."""
        logger.info(
            f"--- Attempt {state['current_attempt'] + 1} of {state['max_attempts']}: Generating SQL ---"
        )
        chat_history_str = "\n".join(
            [f"Human: {q}\nAI: {a}" for q, a in state.get("chat_history", [])]
        ).strip()
        llm_variables = {
            "database_dialect": self.db_service.dialect,
            "schema_definition": state["db_context"]["table_info"],
            "user_question": state["natural_language_question"],
            "history": "\n".join(state["history"]),
            "chat_history": (
                chat_history_str
                if chat_history_str
                else "No previous conversation history."
            ),
        }
        response = self.llm_service.generate_structured(
            variables=llm_variables, response_model=SQLAgentResponse
        )

        history_entry = (
            f"ATTEMPT {state['current_attempt'] + 1} - Status: {response.status}"
        )
        if response.query:
            history_entry += f"\nSQL:\n{response.query}"
        if response.reason:
            history_entry += f"\nReason:\n{response.reason}"
        if response.clarification_question:
            history_entry += f"\nClarification:\n{response.clarification_question}"

        return {
            "generation_result": response,
            "current_attempt": state["current_attempt"] + 1,
            "history": state["history"] + [history_entry],
        }

    def execute_sql_node(self, state: SQLAgentState) -> Dict[str, Any]:
        """Node that executes the SQL only if the generation was successful."""
        generation_result = state["generation_result"]
        if generation_result.status != "success":
            logger.warning(
                f"Skipping SQL execution. Generation status: '{generation_result.status}'."
            )
            # We don't need to return anything here, the final run() method will handle it.
            return {}

        generated_sql = generation_result.query.strip()
        logger.info("--- Executing SQL ---")
        try:
            results_df = self.db_service.execute_for_dataframe(generated_sql)
            logger.info("Successfully executed SQL.")
            return {
                "final_dataframe": results_df,
                "generated_sql": generated_sql,
                "error": None,
            }
        except (ValueError, RuntimeError) as e:
            error_message = f"Error executing SQL: {e}"
            logger.error(f"Execution failed with error:\n{error_message}")
            return {"error": error_message, "final_dataframe": None}

    def should_retry_node(self, state: SQLAgentState) -> str:
        """Decides whether to retry SQL generation or end the workflow."""
        generation_status = state["generation_result"].status
        db_error = state.get("error")

        # If generation status was not 'success', there's nothing to execute or retry.
        if generation_status != "success":
            return "end"

        # If there was no database error, the execution was successful.
        if db_error is None:
            logger.info("--- Workflow successful ---")
            return "end"

        # If there was a DB error but we are out of attempts, end.
        if state["current_attempt"] >= state["max_attempts"]:
            logger.warning("--- Max attempts reached, ending workflow ---")
            return "end"

        # If there was a DB error and we have attempts left, retry.
        logger.info(f"--- Database error found, retrying ---")
        return "retry"

    def run(
        self,
        natural_language_question: str,
        chat_history: Optional[List[Tuple[str, str]]] = None,
    ) -> Dict[str, Any]:
        """
        Executes the agent's workflow and returns a structured status dictionary.
        This is the public interface for the agent.
        """
        initial_state: SQLAgentState = {
            "natural_language_question": natural_language_question,
            "chat_history": chat_history or [],
            "db_context": self.db_service.get_context_for_agent(),
            "history": [],
            "max_attempts": self.max_attempts,
            "current_attempt": 0,
            "generation_result": None,
            "final_dataframe": None,
            "generated_sql": "",
            "error": None,
        }

        final_state = self.app.invoke(initial_state)
        final_result = final_state["generation_result"]

        # No more guesswork. We directly inspect the structured result's status.
        if final_result.status == "success":
            db_error = final_state.get("error")
            if db_error:
                return {
                    "status": "error",
                    "reason": f"SQL execution failed after multiple attempts: {db_error}",
                }
            else:
                return {
                    "status": "success",
                    "dataframe": final_state.get("final_dataframe"),
                    "sql_query": final_state.get("generated_sql"),
                }
        elif final_result.status == "clarification":
            return {
                "status": "clarification",
                "question": final_result.clarification_question,
            }
        else:  # status == "error"
            return {"status": "error", "reason": final_result.reason}

