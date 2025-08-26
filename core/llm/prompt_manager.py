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

    def get_prompts(self, prompts_dir: str) -> Tuple[str, str]:
        """
        Loads the system and user prompt templates for a given agent.

        Args:
            prompts_dir: The name of the agent's prompt directory,
                                    corresponding to a folder in the prompts_base_path.

        Returns:
            A tuple containing the system prompt and user prompt templates.
        """
        agent_prompt_dir = self.prompts_base_path / prompts_dir
        if not agent_prompt_dir.is_dir():
            raise FileNotFoundError(f"Prompt directory for agent '{prompts_dir}' not found at {agent_prompt_dir}")

        system_prompt_path = agent_prompt_dir / "system.prompt"
        user_prompt_path = agent_prompt_dir / "user.prompt"

        system_prompt = self._read_file(system_prompt_path)
        user_prompt = self._read_file(user_prompt_path)

        return system_prompt, user_prompt

    def get_few_shot_examples(self, prompts_dir: str) -> Optional[List[Dict[str, str]]]:
        """
        Loads optional few-shot examples for a given agent from a JSON file.

        The JSON file should be a list of dictionaries, where each dictionary
        represents an example (e.g., {"user": "...", "assistant": "..."}).

        Args:
            prompts_dir: The name of the agent's prompt directory.

        Returns:
            A list of few-shot examples, or None if the file doesn't exist.
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
