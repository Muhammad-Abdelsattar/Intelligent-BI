from typing import Optional, Literal
from pydantic import BaseModel, Field, model_validator

class SQLAgentResponse(BaseModel):
    """A unified model for the LLM's response, covering success and error cases."""
    status: Literal["success", "error"] = Field(
        ...,
        description="Set to 'success' if a query was generated, or 'error' if it was not."
    )
    query: Optional[str] = Field(
        None,
        description="The executable SQL query. This MUST be populated if status is 'success'."
    )
    reason: Optional[str] = Field(
        None,
        description="The reason for failure. This MUST be populated if status is 'error'."
    )

    @model_validator(mode='after')
    def check_query_or_reason_exists(cls, self):
        """Ensures that the correct field is populated based on the status."""
        if self.status == 'success':
            if not self.query:
                raise ValueError("Field 'query' is required when status is 'success'.")
            if self.reason:
                raise ValueError("Field 'reason' must not be set when status is 'success'.")
        elif self.status == 'error':
            if not self.reason:
                raise ValueError("Field 'reason' is required when status is 'error'.")
            if self.query:
                raise ValueError("Field 'query' must not be set when status is 'error'.")
        return self