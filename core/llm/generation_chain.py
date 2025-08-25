from typing import Dict, Any, List, Optional
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, BaseMessage

class GenerationChain:
    """
    A self-contained, reusable chain for generating LLM responses.

    This object encapsulates a specific LLM client and its associated prompt
    templates. Agents will use an instance of this class to perform their
    core generation task repeatedly without needing to re-specify prompts or models.
    """

    def __init__(
        self,
        llm_client: BaseChatModel,
        system_prompt_template: str,
        human_prompt_template: str
    ):
        """
        Initializes the chain with a pre-configured client and raw prompt templates.

        Args:
            llm_client: An initialized LangChain chat model client.
            system_prompt_template: The raw string template for the system prompt.
            human_prompt_template: The raw string template for the human prompt.
        """
        self.llm_client = llm_client
        self.system_prompt_template = system_prompt_template
        self.human_prompt_template = human_prompt_template

    def invoke(
        self,
        variables: Dict[str, Any],
        few_shot_examples: Optional[List[Dict[str, str]]] = None
    ) -> str:
        """
        Executes the generation chain with the provided dynamic variables.

        Args:
            variables: A dictionary of values to format the prompt templates.
            few_shot_examples: An optional list of dictionaries, where each
                               dictionary represents a user/assistant interaction.
                               Expected keys are 'user' and 'assistant'.

        Returns:
            The string content of the AI's response.
        """
        messages: List[BaseMessage] = []

        # 1. System Message
        system_content = self.system_prompt_template.format(**variables)
        messages.append(SystemMessage(content=system_content))

        # 2. Few-shot examples
        if few_shot_examples:
            for example in few_shot_examples:
                if 'user' in example and 'assistant' in example:
                    messages.append(HumanMessage(content=example['user']))
                    messages.append(AIMessage(content=example['assistant']))

        # 3. Human Message
        human_content = self.human_prompt_template.format(**variables)
        messages.append(HumanMessage(content=human_content))

        ai_response = self.llm_client.invoke(messages)

        if not hasattr(ai_response, 'content'):
            response_type = type(ai_response).__name__
            raise TypeError(
                f"The response from the LLM client (type: {response_type}) does not have a 'content' attribute. "
                "Ensure the LLM client returns a standard LangChain message object."
            )
            
        return str(ai_response.content)