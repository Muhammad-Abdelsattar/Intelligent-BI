from typing import Optional, Literal
from pydantic import BaseModel, Field, model_validator


class SQLAgentResponse(BaseModel):
    """A unified model for the LLM's response, covering success, error, and clarification."""

    status: Literal["success", "error", "clarification"] = Field(
        ...,
        description="Set to 'success' if a query was generated, 'error' if it was not possible, or 'clarification' if more information is needed from the user.",
    )
    query: Optional[str] = Field(
        None,
        description="The executable SQL query. This MUST be populated if status is 'success'.",
    )
    reason: Optional[str] = Field(
        None,
        description="The reason for failure. This MUST be populated if status is 'error'.",
    )
    clarification_question: Optional[str] = Field(
        None,
        description="A specific question to ask the user to resolve ambiguity. This MUST be populated if status is 'clarification'.",
    )

    @model_validator(mode="after")
    def check_fields(cls, self):
        """Ensures that the correct field is populated based on the status."""
        if self.status == "success":
            if not self.query:
                raise ValueError("Field 'query' is required when status is 'success'.")
        elif self.status == "error":
            if not self.reason:
                raise ValueError("Field 'reason' is required when status is 'error'.")
        elif self.status == "clarification":
            if not self.clarification_question:
                raise ValueError(
                    "Field 'clarification_question' is required when status is 'clarification'."
                )
        return self

