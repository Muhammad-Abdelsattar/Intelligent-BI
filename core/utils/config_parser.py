import os
from pathlib import Path
from omegaconf import OmegaConf, DictConfig

def load_config(config_path: Path) -> DictConfig:
    """
    Loads a configuration file using OmegaConf and registers the 'env' resolver.

    Args:
        config_path: The path to the configuration file.

    Returns:
        An OmegaConf DictConfig object.

    Raises:
        FileNotFoundError: If the configuration file does not exist.
        Exception: For other errors during configuration loading.
    """
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")

    # Register the 'env' resolver if not already registered
    # This check prevents re-registration errors if called multiple times
    if not OmegaConf.has_resolver("env"):
        OmegaConf.register_resolver("env", lambda name: os.environ.get(name))

    try:
        config = OmegaConf.load(config_path)
        return config
    except Exception as e:
        raise Exception(f"Error loading configuration from {config_path}: {e}")
