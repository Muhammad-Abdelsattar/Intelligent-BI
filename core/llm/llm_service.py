from typing import Dict, Any, List, Type
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
    """

    def __init__(self, agent_prompts_dir: str, provider_key: str, llm_config: DictConfig, prompts_base_path: Path):
        """Initializes the complete LLM stack for a specific agent."""
        llm_factory = LLMFactory(llm_config=llm_config)
        self.llm_client = llm_factory.create_llm_client(provider_key)
        prompt_manager = PromptManager(prompts_base_path=prompts_base_path)
        self.system_prompt_template, self.human_prompt_template = prompt_manager.get_prompts(agent_prompts_dir)
        self.few_shot_examples = prompt_manager.get_few_shot_examples(agent_prompts_dir)

    def _build_messages(self, variables: Dict[str, Any]) -> List[BaseMessage]:
        """Helper to build the list of messages for the LLM."""
        messages: List[BaseMessage] = []
        system_content = self.system_prompt_template.format(**variables)
        messages.append(SystemMessage(content=system_content))
        if self.few_shot_examples:
            for example in self.few_shot_examples:
                if 'user' in example and 'assistant' in example:
                    messages.append(HumanMessage(content=example['user']))
                    messages.append(AIMessage(content=example['assistant']))
        human_content = self.human_prompt_template.format(**variables)
        messages.append(HumanMessage(content=human_content))
        return messages

    def generate_text(self, variables: Dict[str, Any]) -> str:
        """Generates a raw text response from the LLM."""
        messages = self._build_messages(variables)
        response = self.llm_client.invoke(messages)
        return response.content

    def generate_structured(self, variables: Dict[str, Any], response_model: Type[BaseModel]) -> BaseModel:
        """
        Generates a structured response from the LLM, parsed into a Pydantic model.
        """
        messages = self._build_messages(variables)
        # With the unified model, this method is simple and direct.
        structured_llm = self.llm_client.with_structured_output(response_model)
        structured_response = structured_llm.invoke(messages)
        return structured_response