from typing import TypedDict, List, Any, Dict, Optional
import pandas as pd


class OrchestratorState(TypedDict):
    """
    Represents the state of the main orchestration workflow.
    This is the single source of truth that is passed between nodes in the graph.
    """

    # -- Inputs --
    user_prompt: str
    chat_history: List[tuple[str, str]]

    # -- Agent Outputs / Artifacts --
    sql_query: Optional[str]
    dataframe: Optional[pd.DataFrame]
    analysis_text: Optional[str]
    chart_config: Optional[Dict[str, Any]]

    # -- Control Flow & Communication --
    next_action: Optional[Any]
    # Holds a question from an expert agent (like SQLAgent) for the user.
    clarification_question: Optional[str]
    error: Optional[str]

    # -- Final Output --
    # The final, user-facing result is packaged here.
    output: Optional[Dict[str, Any]]
