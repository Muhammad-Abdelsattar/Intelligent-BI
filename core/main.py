import os
from omegaconf import OmegaConf

# Register the 'env' resolver for OmegaConf to resolve environment variables
OmegaConf.register_resolver("env", lambda name: os.environ.get(name))

# Example of loading the database configuration
if __name__ == "__main__":
    # Set some dummy environment variables for testing
    os.environ["DB_HOST"] = "localhost"
    os.environ["DB_PORT"] = "5432"
    os.environ["DB_USER"] = "test_user"
    os.environ["DB_PASSWORD"] = "test_password"
    os.environ["DB_NAME"] = "test_db"

    try:
        # Load the configuration from db_config.yaml
        config = OmegaConf.load("core/config/db_config.yaml")

        # Access the database configuration
        db_config = config.database

        print("Successfully loaded database configuration:")
        print(OmegaConf.to_yaml(db_config))

        # You can now use db_config to initialize your DatabaseService
        # from core.database import DatabaseService
        # db_service = DatabaseService(db_config)
        # print(f"Database URI: {db_service._manager.get_uri()}")

    except Exception as e:
        print(f"Error loading configuration: {e}")
        print("Please ensure environment variables are set and db_config.yaml exists.")
