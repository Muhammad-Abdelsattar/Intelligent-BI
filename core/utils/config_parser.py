import os
from pathlib import Path

from omegaconf import DictConfig, OmegaConf


def find_project_root(marker: str = "pyproject.toml") -> Path:
    """
    Finds the project root by searching upwards for a marker file.

    This function starts from the current file's location and traverses up
    the directory tree until it finds a directory containing the specified
    marker file.

    Args:
        marker: The name of the marker file to find (e.g., 'pyproject.toml').

    Returns:
        The Path object representing the project root directory.

    Raises:
        FileNotFoundError: If the project root cannot be found by traversing
                           up from the current file's location.
    """
    current_path = Path(__file__).resolve()
    while current_path != current_path.parent:  # Stop at the filesystem root
        if (current_path / marker).exists():
            return current_path
        current_path = current_path.parent

    raise FileNotFoundError(
        f"Could not find the project root. "
        f"Searched for a '{marker}' file from '{Path(__file__).resolve()}' upwards."
    )


# --- Application-wide Constants ---

# Define the project root as a constant to be used throughout the application
PROJECT_ROOT = find_project_root()


def load_app_config(config_dir: str = "core/config") -> DictConfig:
    """
    Loads all YAML configuration files from a directory into a single,
    namespaced OmegaConf DictConfig object.

    Each YAML file is loaded under a key corresponding to its filename stem.
    For example, `databases.yaml` will be accessible under the `databases` key
    in the returned config object. This prevents key collisions between different
    configuration files.

    It also registers a resolver to read environment variables with `${env:VAR_NAME}`.

    Args:
        config_dir: The relative path to the configuration directory from the
                    project root.

    Returns:
        A single, merged OmegaConf DictConfig object containing all configurations.

    Raises:
        FileNotFoundError: If the specified configuration directory does not exist.
    """
    # Use the modern, non-deprecated API for registering resolvers.
    # This check prevents errors if the function is called multiple times.
    if not OmegaConf.has_resolver("env"):
        OmegaConf.register_new_resolver("env", lambda name: os.environ.get(name))

    config_path = PROJECT_ROOT / config_dir
    if not config_path.is_dir():
        raise FileNotFoundError(f"Configuration directory not found at '{config_path.resolve()}'")

    # Create a base config object to merge into
    merged_config = OmegaConf.create()

    # Load each YAML file under a key corresponding to its filename
    for p in config_path.glob("*.yaml"):
        key = p.stem  # 'agents.yaml' -> 'agents'
        try:
            conf = OmegaConf.load(p)
            merged_config[key] = conf
        except Exception as e:
            # Provide more context on which file failed to load
            raise RuntimeError(f"Failed to load or parse configuration file '{p.name}': {e}") from e

    return merged_config