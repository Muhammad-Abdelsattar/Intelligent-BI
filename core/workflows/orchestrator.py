from typing import Dict, Any, List, Tuple
from omegaconf import DictConfig
from pathlib import Path
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage
import pandas as pd

from core.agents.sql_agent import SQLAgent
from core.llm.llm_factory import LLMFactory
from core.workflows.state import OrchestratorState
from core.workflows.tools import ROUTER_TOOLS


class MainOrchestrator:
    """
    The main orchestrator for the BI agent system, built using LangGraph.

    This class manages a stateful graph that routes user prompts to the
    appropriate agent (tool) based on an LLM's decision. The tools are
    defined as Pydantic models for clarity and type safety.
    """

    def __init__(self, app_config: DictConfig, prompts_base_path: Path):
        self.app_config = app_config
        self.prompts_base_path = prompts_base_path
        self.sql_agent = SQLAgent.from_config(
            agent_key="sql_agent",
            app_config=app_config,
            prompts_base_path=prompts_base_path,
        )
        llm_factory = LLMFactory(llm_config=app_config.llms)
        llm = llm_factory.create_llm_client(provider_key="google-gemini-2.5-flash")

        # Binding Pydantic models directly is the cleanest approach.
        self.tool_calling_llm = llm.bind_tools(ROUTER_TOOLS)

        self.workflow = self._build_graph()
        self.app = self.workflow.compile()

    def _build_graph(self) -> StateGraph:
        """Builds the LangGraph workflow."""
        graph = StateGraph(OrchestratorState)

        graph.add_node("router", self.router_node)
        graph.add_node("sql_agent", self.run_sql_agent_node)
        graph.add_node("analysis_agent", self.run_analysis_agent_node)
        graph.add_node("chart_agent", self.run_chart_agent_node)
        graph.add_node("clarification_node", self.run_clarification_node)
        graph.add_node("error_node", self.error_node)
        graph.add_node("finish", self.finish_node)

        graph.set_entry_point("router")

        ### KEY CHANGE: The keys in this map are now the string names of the Pydantic classes.
        graph.add_conditional_edges(
            "router",
            self.decide_next_step,
            {
                "RunSqlAgent": "sql_agent",
                "RunAnalysisAgent": "analysis_agent",
                "RunChartAgent": "chart_agent",
                "AskClarifyingQuestion": "clarification_node",
                "FinishWorkflow": "finish",
            },
        )

        graph.add_conditional_edges(
            "sql_agent",
            self.decide_after_sql,
            {
                "continue": "router",
                "clarify": "clarification_node",
                "error": "error_node",
            },
        )

        graph.add_edge("analysis_agent", "router")
        graph.add_edge("chart_agent", "router")
        graph.add_edge("clarification_node", END)
        graph.add_edge("error_node", END)

        return graph

    def router_node(self, state: OrchestratorState) -> Dict[str, Any]:
        """The brain of the orchestrator. It uses the LLM to decide the next action."""
        print("--- ROUTER ---")
        df_summary = "No"
        if state.get("dataframe") is not None:
            df = state["dataframe"]
            df_summary = f"Yes, with {len(df)} rows and columns: {list(df.columns)}"

        prompt = f"""
        You are an expert AI project manager. Based on the user's request and the current state of the project, decide which tool to use next.

        User's original request: "{state['user_prompt']}"

        Current State:
        - Data Retrieved: {df_summary}
        - Analysis Performed: {'Yes' if state.get('analysis_text') else 'No'}
        - Chart Generated: {'Yes' if state.get('chart_config') else 'No'}

        Your only job is to choose the next tool to call. If the user's request has been fully satisfied, call the 'FinishWorkflow' tool.
        """
        messages = [HumanMessage(content=prompt)]
        response = self.tool_calling_llm.invoke(messages)
        return {"next_action": response.tool_calls[0]}

    def run_sql_agent_node(self, state: OrchestratorState) -> Dict[str, Any]:
        """Executes the SQL Agent and routes based on its explicit status report."""
        print("--- RUNNING SQL AGENT ---")
        result = self.sql_agent.run(
            natural_language_question=state["user_prompt"],
            chat_history=state["chat_history"],
        )
        if result["status"] == "success":
            return {
                "dataframe": result.get("dataframe"),
                "sql_query": result.get("sql_query"),
            }
        elif result["status"] == "clarification":
            return {"clarification_question": result.get("question")}
        else:
            return {"error": result.get("reason")}

    def run_analysis_agent_node(self, state: OrchestratorState) -> Dict[str, Any]:
        print("--- (SIMULATED) RUNNING ANALYSIS AGENT ---")
        return {"analysis_text": "This is a placeholder analysis."}

    def run_chart_agent_node(self, state: OrchestratorState) -> Dict[str, Any]:
        print("--- (SIMULATED) RUNNING CHART AGENT ---")
        return {"chart_config": {"type": "bar", "title": "Placeholder Chart"}}

    def run_clarification_node(self, state: OrchestratorState) -> Dict[str, Any]:
        """Prepares a clarifying question as the final output for this turn."""
        print("--- HANDLING CLARIFICATION ---")
        question = (
            state.get("clarification_question")
            or state["next_action"]["args"]["question"]
        )
        return {"output": {"content": question, "is_clarification": True}}

    def error_node(self, state: OrchestratorState) -> Dict[str, Any]:
        """Packages an agent error as the final output."""
        print("--- HANDLING AGENT ERROR ---")
        error_message = state.get("error", "An unspecified error occurred.")
        return {"output": {"error": error_message}}

    def finish_node(self, state: OrchestratorState) -> Dict[str, Any]:
        """Prepares the final output for the user."""
        print("--- FINISHING WORKFLOW ---")
        final_answer = state["next_action"]["args"].get(
            "answer", "I have completed the request."
        )
        return {
            "output": {
                "content": final_answer,
                "dataframe": state.get("dataframe"),
                "sql_query": state.get("sql_query"),
                "analysis_text": state.get("analysis_text"),
                "chart_config": state.get("chart_config"),
            }
        }

    def decide_next_step(self, state: OrchestratorState) -> str:
        """The name of the Pydantic class is the tool name."""
        tool_name = state["next_action"]["name"]
        print(f"--- ROUTER DECISION: {tool_name} ---")
        return tool_name

    def decide_after_sql(self, state: OrchestratorState) -> str:
        """Determines the path after the SQL agent has run."""
        if state.get("error"):
            return "error"
        elif state.get("clarification_question"):
            return "clarify"
        else:
            return "continue"

    def run(self, prompt: str, chat_history: List[Tuple[str, str]]) -> Dict[str, Any]:
        initial_state = {
            "user_prompt": prompt,
            "chat_history": chat_history,
            "dataframe": None,
            "sql_query": None,
            "analysis_text": None,
            "chart_config": None,
            "output": None,
            "error": None,
            "next_action": None,
            "clarification_question": None,
        }
        final_state = self.app.invoke(initial_state)
        return final_state.get("output", {"error": "The workflow ended unexpectedly."})
