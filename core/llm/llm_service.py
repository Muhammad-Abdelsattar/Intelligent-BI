from typing import Dict, Any

from core.llm.llm_factory import LLMFactory
from core.llm.prompt_manager import PromptManager
from core.llm.generation_chain import GenerationChain

class LLMService:
    """
    A high-level interface for AI agents to interact with the LLM.

    This class abstracts away the details of prompt management, LLM client
    creation, and the generation chain. An agent only needs to instantiate
    this class with its name and a chosen LLM provider, and then call
    the `generate_response` method.
    """

    def __init__(self, agent_name: str, provider_key: str, llm_config_path: str = "core/config/llm_providers.yaml", prompts_base_path: str = "core/prompts"):
        """
        Initializes the complete LLM stack for a specific agent.

        Args:
            agent_name: The name of the agent (e.g., 'sql_generator').
            provider_key: The key for the LLM provider from the config
                          (e.g., 'azure_openai_4o').
            llm_config_path: Path to the LLM providers configuration file.
            prompts_base_path: Path to the prompts directory.
        """
        # 1. Create LLM client
        llm_factory = LLMFactory(config_path=llm_config_path)
        llm_client = llm_factory.create_llm_client(provider_key)

        # 2. Load prompts and examples
        prompt_manager = PromptManager(prompts_base_path=prompts_base_path)
        system_prompt, user_prompt = prompt_manager.get_prompts(agent_name)
        self.few_shot_examples = prompt_manager.get_few_shot_examples(agent_name)

        # 3. Create Generation Chain
        self._chain = GenerationChain(
            llm_client=llm_client,
            system_prompt_template=system_prompt,
            human_prompt_template=user_prompt
        )

    def _extract_sql_from_response(self, raw_response: str) -> str:
        """
        Extracts the SQL query from a markdown code block in the raw LLM response.
        Assumes the SQL is within ```sql ... ```.
        """
        start_tag = "```sql"
        end_tag = "```"

        start_index = raw_response.find(start_tag)
        if start_index == -1:
            # If no start tag, check for error message
            if raw_response.strip().startswith("Error:"):
                return raw_response # Return error message as is
            raise ValueError("LLM response does not contain a '```sql' block.")

        # Adjust start_index to point to the character after the start_tag
        start_index += len(start_tag)

        end_index = raw_response.find(end_tag, start_index)
        if end_index == -1:
            raise ValueError("LLM response contains '```sql' but no closing '```' tag.")

        extracted_sql = raw_response[start_index:end_index].strip()
        return extracted_sql

    def generate_response(self, variables: Dict[str, Any]) -> str:
        """
        Generates a response from the LLM using the pre-configured chain
        and extracts the SQL query from the markdown block.

        Args:
            variables: A dictionary of dynamic values to format the prompt
                       templates (e.g., {'user_question': '...', 'schema': '...'}).

        Returns:
            The extracted SQL query string, or an error message if the LLM
            returned one.
        """
        raw_response = self._chain.invoke(
            variables=variables,
            few_shot_examples=self.few_shot_examples
        )
        return self._extract_sql_from_response(raw_response)
