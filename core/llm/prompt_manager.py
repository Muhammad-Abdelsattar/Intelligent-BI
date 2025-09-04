import json
from pathlib import Path
from typing import Tuple, List, Dict, Optional

class PromptManager:
    """
    Manages loading and providing prompt templates for different agents
    using pathlib for path operations.
    """

    def __init__(self, prompts_base_path: Path):
        """
        Initializes the PromptManager with the base path for prompts.

        Args:
            prompts_base_path: The root directory where agent-specific prompt
                               folders are located.
        """
        if not prompts_base_path.is_dir():
            raise FileNotFoundError(f"Prompts base directory not found at: {prompts_base_path}")
        self.prompts_base_path = prompts_base_path

    def _read_file(self, path: Path) -> str:
        """Reads a file and returns its content."""
        try:
            return path.read_text(encoding='utf-8')
        except FileNotFoundError:
            raise FileNotFoundError(f"Prompt file not found at: {path}")
        except Exception as e:
            raise IOError(f"Error reading prompt file at {path}: {e}")

    def load_prompt(self, prompts_dir: str, filename: str) -> str:
        """
        Loads a single, specific prompt file from an agent's prompt directory.

        This is the core, generic method for loading any prompt file.

        Args:
            prompts_dir: The name of the agent's prompt directory.
            filename: The name of the file to load (e.g., 'system.prompt').

        Returns:
            The content of the prompt file as a string.
        """
        agent_prompt_dir = self.prompts_base_path / prompts_dir
        if not agent_prompt_dir.is_dir():
            raise FileNotFoundError(f"Prompt directory for agent '{prompts_dir}' not found at {agent_prompt_dir}")
        
        prompt_path = agent_prompt_dir / filename
        return self._read_file(prompt_path)

    def get_standard_prompts(
        self,
        prompts_dir: str,
        system_filename: str = "system.prompt",
        user_filename: str = "user.prompt"
    ) -> Tuple[str, Optional[str]]:
        """
        Loads the standard system and an optional user prompt template.

        If the user prompt file is not found, it returns None for it,
        allowing for agents that only use a system prompt by default.

        Args:
            prompts_dir: The name of the agent's prompt directory.
            system_filename: The filename for the system prompt.
            user_filename: The filename for the user prompt.

        Returns:
            A tuple containing the system prompt string and the user prompt string or None.
        """
        system_prompt = self.load_prompt(prompts_dir, system_filename)
        user_prompt = None
        try:
            user_prompt = self.load_prompt(prompts_dir, user_filename)
        except FileNotFoundError:
            # It's acceptable for an agent to not have a default user prompt
            pass
            
        return system_prompt, user_prompt

    def get_few_shot_examples(self, prompts_dir: str) -> Optional[List[Dict[str, str]]]:
        """
        Loads optional few-shot examples for a given agent from a JSON file.
        (This method remains unchanged but could be refactored similarly if needed).
        """
        examples_path = self.prompts_base_path / prompts_dir / "few_shot_examples.json"
        if not examples_path.exists():
            return None

        try:
            with examples_path.open('r', encoding='utf-8') as f:
                examples = json.load(f)
            if not isinstance(examples, list):
                raise ValueError("Few-shot examples file must contain a JSON list.")
            return examples
        except json.JSONDecodeError as e:
            raise ValueError(f"Error decoding JSON from {examples_path}: {e}")
        except Exception as e:
            raise IOError(f"Error reading few-shot examples file at {examples_path}: {e}")