from typing import List, Literal, TypedDict

class ChatMessage(TypedDict):
    """Represents a single message in the conversation history."""
    role: Literal["human", "ai"]
    content: str

class ConversationMemoryState(TypedDict):
    """
    Represents the complete, canonical state of a conversation's memory.
    This is the "source of truth" managed by the ConversationMemoryService.
    """
    message_buffer: List[ChatMessage]
    summary: str

class WorkingContext(TypedDict):
    """
    Represents the lean, curated context passed to a worker agent.
    This is the output of the context formulation process.
    """
    chat_history: List[tuple[str, str]] # The (human, ai) format agents expect
    summary: str