import logging
from omegaconf import DictConfig
from pathlib import Path
from typing import List, Literal

from core.llm import LLMService
from core.memory.state import ChatMessage, ConversationMemoryState, WorkingContext

logger = logging.getLogger(__name__)

class ConversationMemoryService:
    """
    A stateful service that manages the lifecycle of a single conversation's memory
    using a summarization-threshold strategy.
    """

    def __init__(
        self,
        llm_config: DictConfig,
        prompts_base_path: Path,
        summarizer_llm_key: str,
        max_buffer_size: int = 10
    ):
        """
        Initializes the memory service.

        Args:
            llm_config: The global LLM configuration.
            prompts_base_path: The path to the root prompts directory.
            summarizer_llm_key: The key for the fast, cheap LLM to use for summarization.
            max_buffer_size: The number of messages (human + ai) to hold before triggering summarization.
        """
        if max_buffer_size % 2 != 0:
            logger.warning("max_buffer_size should ideally be an even number to capture full Q&A pairs.")
        
        self._state: ConversationMemoryState = {
            "message_buffer": [],
            "summary": "This is the beginning of the conversation."
        }
        self.max_buffer_size = max_buffer_size
        
        # This service has its own internal LLMService dedicated to summarization
        self._summarizer_llm = LLMService(
            agent_prompts_dir="summarizer",
            provider_key=summarizer_llm_key,
            llm_config=llm_config,
            prompts_base_path=prompts_base_path
        )
        logger.info(f"MemoryService initialized with buffer size {max_buffer_size}.")

    def _trigger_summarization(self):
        """
        Uses an LLM to summarize the current buffer and updates the internal state.
        """
        if not self._state["message_buffer"]:
            return

        logger.info(f"Buffer limit of {self.max_buffer_size} reached. Triggering summarization.")
        
        # Format the buffer into a single string for the prompt
        conversation_text = "\n".join(
            [f"{msg['role']}: {msg['content']}" for msg in self._state["message_buffer"]]
        )

        variables = {
            "previous_summary": self._state["summary"],
            "conversation_text": conversation_text
        }

        # Generate the new summary
        new_summary = self._summarizer_llm.generate_text(variables)

        # Update the state
        self._state["summary"] = new_summary
        self._state["message_buffer"] = [] # Clear the buffer
        logger.info(f"Summarization complete. New summary:\n{new_summary}")

    def add_message(self, role: Literal["human", "ai"], content: str):
        """
        Adds a new message to the memory buffer and triggers summarization if the
        buffer size threshold is met.
        """
        if role not in ["human", "ai"]:
            raise ValueError("Role must be either 'human' or 'ai'.")
        
        self._state["message_buffer"].append({"role": role, "content": content})

        # Check if it's time to summarize
        if len(self._state["message_buffer"]) >= self.max_buffer_size:
            self._trigger_summarization()

    def get_context_for_agent(self) -> WorkingContext:
        """
        Constructs and returns the working context for an agent, combining the
        current summary and the message buffer.
        """
        # Convert the current buffer to the (human, ai) tuple format
        buffer_tuples = []
        human_msg = None
        for msg in self._state["message_buffer"]:
            if msg["role"] == "human":
                human_msg = msg["content"]
            elif msg["role"] == "ai" and human_msg is not None:
                buffer_tuples.append((human_msg, msg["content"]))
                human_msg = None # Reset after pairing

        return {
            "summary": self._state["summary"],
            "chat_history": buffer_tuples
        }