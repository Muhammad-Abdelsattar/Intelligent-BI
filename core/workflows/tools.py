from pydantic import BaseModel, Field
from typing import List

class RunSqlAgent(BaseModel):
    """
    Retrieves data from the database based on a user's question.
    Use this to get the data needed to answer the user's request.
    This should almost always be the first step.
    """
    user_prompt: str = Field(..., description="The original, natural language question from the user.")

class RunAnalysisAgent(BaseModel):
    """
    Analyzes the retrieved data to generate textual insights.
    Use this AFTER 'RunSqlAgent' has been successfully executed.
    """
    data_summary: str = Field(..., description="A brief summary of the data that has been retrieved, including row count and column names.")

class RunChartAgent(BaseModel):
    """
    Generates a chart configuration for visualizing the data.
    Use this AFTER 'RunSqlAgent' has been successfully executed.
    Only use if the user has explicitly asked for a 'chart', 'plot', or 'visualization'.
    """
    data_summary: str = Field(..., description="A brief summary of the data to be visualized, including row count and column names.")

class AskClarifyingQuestion(BaseModel):
    """
    Asks the user a question to clarify their request.
    Use this when the user's prompt is ambiguous, vague, or missing
    critical information needed to proceed.
    """
    question: str = Field(..., description="The specific, concise question to ask the user to resolve the ambiguity.")

class FinishWorkflow(BaseModel):
    """
    Completes the workflow and provides a final answer to the user.
    Use this ONLY when all necessary steps have been taken and you have
    all the information (data, analysis, charts) needed to fully
    address the user's original request.
    """
    answer: str = Field(..., description="The final, comprehensive answer to the user's question, summarizing the findings.")


# A list of all Pydantic models that the router can choose from.
# LangChain will automatically convert these into tool definitions for the LLM.
ROUTER_TOOLS: List[BaseModel] = [
    RunSqlAgent,
    RunAnalysisAgent,
    RunChartAgent,
    AskClarifyingQuestion,
    FinishWorkflow,
]
