import os
import logging
import sys
from omegaconf import OmegaConf, DictConfig
from pathlib import Path

# --- Logging Setup ---
# Configure logging to output to stdout for better visibility in containerized environments
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

# --- Configuration Loading ---
OmegaConf.register_resolver("env", lambda name: os.environ.get(name))

def load_app_config() -> DictConfig:
    """Loads all configuration files from the core/config directory."""
    config_path = Path("core/config")
    if not config_path.is_dir():
        logger.error(f"Configuration directory not found at '{config_path.resolve()}'")
        raise FileNotFoundError(f"Configuration directory not found at '{config_path.resolve()}'")

    # Load all .yaml files in the config directory
    configs = [OmegaConf.load(p) for p in config_path.glob("*.yaml")]
    # Merge them into a single configuration object
    # Note: If keys overlap, the file loaded last will take precedence.
    return OmegaConf.merge(*configs)

# --- Main Application Logic ---
if __name__ == "__main__":
    logger.info("Application starting up...")
    try:
        # Load the application configuration
        app_config = load_app_config()
        logger.info("Configuration loaded successfully.")
        # Example: Accessing a specific agent's config
        # sql_agent_config = app_config.sql_agent
        # logger.info(f"SQL Agent Config:\n{OmegaConf.to_yaml(sql_agent_config)}")

        # Here you would initialize your main application components/services
        # For example, creating an agent instance:
        # from core.agents import SQLAgent
        # sql_agent = SQLAgent.from_config(
        #     agent_config=app_config.sql_agent,
        #     databases_config=app_config, # Pass the whole config for key-based access
        #     llm_providers_config=app_config,
        #     prompts_base_path=Path("core/prompts")
        # )
        # logger.info("SQLAgent created successfully.")

    except Exception as e:
        logger.critical(f"A critical error occurred during startup: {e}", exc_info=True)
        sys.exit(1) # Exit with a non-zero code to indicate failure