import importlib
from typing import Dict, Any

from omegaconf import OmegaConf, DictConfig
from langchain_core.language_models.chat_models import BaseChatModel

class LLMFactory:
    """
    A factory class for creating LangChain LLM clients using OmegaConf.
    """

    def __init__(self, config_path: str = "core/config/llm_providers.yaml"):
        """
        Initializes the factory by loading the LLM provider configuration.

        Args:
            config_path: The path to the OmegaConf YAML file.
        """
        self._config = self._load_config(config_path)

    def _load_config(self, config_path: str) -> DictConfig:
        """Loads the model configuration from the YAML file using OmegaConf."""
        try:
            conf = OmegaConf.load(config_path)
            if not isinstance(conf, DictConfig) or 'llm_providers' not in conf or not isinstance(conf.llm_providers, DictConfig):
                raise ValueError("YAML config must be a dictionary and contain a 'llm_providers' dictionary.")
            return conf.llm_providers
        except FileNotFoundError:
            raise FileNotFoundError(f"LLM configuration file not found at: {config_path}")
        except Exception as e:
            raise RuntimeError(f"Error parsing LLM configuration with OmegaConf from '{config_path}': {e}")

    def get_available_providers(self) -> Dict[str, str]:
        """
        Returns a dictionary of available provider keys and their display names.
        Reads from the pre-loaded OmegaConf configuration object.
        """
        if not self._config:
            return {}
        return {
            key: provider.get('display_name', key)
            for key, provider in self._config.items()
            if provider
        }

    def create_llm_client(self, provider_key: str) -> BaseChatModel:
        """
        Creates an LLM client instance based on the provider key.

        Args:
            provider_key: The key from the llm_providers.yaml file (e.g., 'azure_openai_4o').

        Returns:
            An instance of the specified LangChain LLM client.
        """
        if not self._config or provider_key not in self._config:
            raise ValueError(f"Provider '{provider_key}' not found in the configuration. "
                             f"Available providers: {list(self._config.keys() if self._config else [])}")

        provider_config = self._config.get(provider_key)

        if 'class' not in provider_config or 'params' not in provider_config:
            raise ValueError(f"Provider '{provider_key}' configuration is missing 'class' or 'params'.")

        resolved_params = OmegaConf.to_container(provider_config.params, resolve=True)

        module_path, class_name = provider_config['class'].rsplit('.', 1)
        try:
            module = importlib.import_module(module_path)
            llm_class = getattr(module, class_name)
        except ImportError:
            raise ImportError(f"Could not import module '{module_path}' for LLM provider '{provider_key}'.")
        except AttributeError:
            raise AttributeError(f"Could not find class '{class_name}' in module '{module_path}'.")
        except Exception as e:
            raise RuntimeError(f"An unexpected error occurred while loading the LLM class: {e}")

        try:
            return llm_class(**resolved_params)
        except TypeError as e:
            raise TypeError(f"Failed to instantiate LLM client for '{provider_key}'. "
                            f"Check if the parameters in the config match the class constructor. Error: {e}")