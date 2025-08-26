from typing import Dict, Any, List, Optional, Type, Union
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, BaseMessage
from pydantic import BaseModel
from omegaconf import DictConfig
from pathlib import Path

from core.llm.llm_factory import LLMFactory
from core.llm.prompt_manager import PromptManager

class LLMService:
    """
    A high-level interface for AI agents to interact with the LLM.

    This class abstracts away the details of prompt management and LLM client
    creation, providing methods for different types of generation (text, structured).
    """

    def __init__(self, agent_prompts_dir: str, provider_key: str, llm_config: DictConfig, prompts_base_path: Path):
        """
        Initializes the complete LLM stack for a specific agent.

        Args:
            agent_prompts_dir: The name of the agent's prompt directory (e.g., 'sql_generator').
            provider_key: The key for the LLM provider from the config
                          (e.g., 'azure_openai_4o').
            llm_config: OmegaConf DictConfig containing the entire 'llms.yaml' content.
            prompts_base_path: Path to the prompts directory.
        """
        # 1. Create LLM client
        llm_factory = LLMFactory(llm_config=llm_config)
        self.llm_client = llm_factory.create_llm_client(provider_key)

        # 2. Load prompts and examples
        prompt_manager = PromptManager(prompts_base_path=prompts_base_path)
        self.system_prompt_template, self.human_prompt_template = prompt_manager.get_prompts(agent_prompts_dir)
        self.few_shot_examples = prompt_manager.get_few_shot_examples(agent_prompts_dir)

    def _build_messages(self, variables: Dict[str, Any]) -> List[BaseMessage]:
        """Helper to build the list of messages for the LLM."""
        messages: List[BaseMessage] = []

        # 1. System Message
        system_content = self.system_prompt_template.format(**variables)
        messages.append(SystemMessage(content=system_content))

        # 2. Few-shot examples
        if self.few_shot_examples:
            for example in self.few_shot_examples:
                if 'user' in example and 'assistant' in example:
                    messages.append(HumanMessage(content=example['user']))
                    messages.append(AIMessage(content=example['assistant']))

        # 3. Human Message
        human_content = self.human_prompt_template.format(**variables)
        messages.append(HumanMessage(content=human_content))
        return messages

    def _extract_sql_from_response(self, raw_response: BaseMessage) -> str:
        """
        Extracts the SQL query from a markdown code block in the raw LLM response.
        Assumes the SQL is within ```sql ... ```.
        """
        start_tag = "```sql"
        end_tag = "```"
        raw_content = raw_response.content

        start_index = raw_content.find(start_tag)
        if start_index == -1:
            # If no start tag, check for error message
            if raw_content.strip().startswith("Error:"):
                return raw_content # Return error message as is
            # If no ```sql block is found, assume the entire response is the SQL.
            # This adds robustness for models that forget the markdown.
            return raw_content.strip()


        # Adjust start_index to point to the character after the start_tag
        start_index += len(start_tag)

        end_index = raw_content.find(end_tag, start_index)
        if end_index == -1:
            # If there's a start but no end, take everything after the start.
            return raw_content[start_index:].strip()

        extracted_sql = raw_content[start_index:end_index].strip()
        return extracted_sql

    def generate_text(self, variables: Dict[str, Any]) -> str:
        """
        Generates a text response from the LLM.
        """
        messages = self._build_messages(variables)
        raw_response = self.llm_client.invoke(messages)
        return self._extract_sql_from_response(raw_response)

    def generate_structured(self, variables: Dict[str, Any], response_model: Type[BaseModel]) -> BaseModel:
        """
        Generates a structured response from the LLM, parsed into a Pydantic model.
        """
        messages = self._build_messages(variables)
        structured_llm = self.llm_client.with_structured_output(response_model)
        structured_response = structured_llm.invoke(messages)
        return structured_response