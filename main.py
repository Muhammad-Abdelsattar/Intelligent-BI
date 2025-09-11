import sys
import logging
from pathlib import Path
from dotenv import load_dotenv
import pandas as pd
from pprint import pprint

# --- PATH SETUP ---
# Add the project root to the Python path to allow imports from 'core'
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

from core.workflows.orchestrator import MainOrchestrator
from core.utils.config_parser import load_app_config, PROJECT_ROOT

# --- LOGGING AND ENVIRONMENT SETUP ---
# Set a basic logging level to see LangGraph's verbose output
logging.basicConfig(level=logging.INFO)
# Suppress excessively noisy logs from underlying HTTP libraries for cleaner output
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)

load_dotenv(PROJECT_ROOT / ".env")


def print_welcome_message():
    """Prints the welcome message and instructions."""
    print("\n--- BI Agent Orchestrator: CLI Test Harness ---")
    print("Type 'exit' or 'quit' to end the session.")
    print("Example prompts to test different workflows:")
    print('  - Simple SQL: "How many customers are there?"')
    print('  - Multi-Step: "Analyze and plot the sales by product category."')
    print('  - Router Clarification: "What about our top performers?"')
    print(
        '  - Agent Clarification: "Show me sales by customer_name." (assuming name column is different)'
    )
    print("----------------------------------------------------------\n")


def display_results(result: dict):
    """Prints all artifacts from the orchestrator's final output in a structured way."""
    print("\n" + "=" * 15 + " Orchestrator Final Output " + "=" * 15)

    if result.get("error"):
        print(f"\n>> 🔴 AI Response (Error): {result['error']}")

    elif result.get("is_clarification"):
        print(f"\n>> 🟡 AI Response (Needs Clarification): {result.get('content')}")

    else:
        # This is the full, successful response
        print(f"\n>> 🟢 AI Response: {result.get('content')}")

        if result.get("sql_query"):
            print("\n--- Generated SQL ---")
            print(result["sql_query"])

        if result.get("analysis_text"):
            print("\n--- Generated Analysis (Simulated) ---")
            print(result["analysis_text"])

        if result.get("chart_config"):
            print("\n--- Generated Chart Config (Simulated) ---")
            pprint(result["chart_config"])

        if result.get("dataframe") is not None:
            print("\n--- Retrieved DataFrame ---")
            df = result["dataframe"]
            if df.empty:
                print("[Empty DataFrame]")
            else:
                # Use to_markdown for a clean table display in the CLI
                print(df.to_markdown(index=False, tablefmt="grid"))

    print("\n" + "=" * 51 + "\n")


def main():
    """Main function to run the interactive CLI test."""
    print("Initializing Orchestrator (this may take a moment)...")
    try:
        app_config = load_app_config()
        prompts_path = PROJECT_ROOT / "core" / "prompts"
        orchestrator = MainOrchestrator(
            app_config=app_config, prompts_base_path=prompts_path
        )
    except Exception as e:
        logging.critical("Failed to initialize orchestrator", exc_info=True)
        print(f"\nFATAL: Could not initialize the system. Error: {e}")
        return

    print_welcome_message()

    chat_history = []

    while True:
        try:
            prompt = input("You: ")
            if prompt.lower() in ["exit", "quit"]:
                print("\nGoodbye!")
                break

            print(
                "\n...Orchestrator is working (see logs above for step-by-step execution)...\n"
            )

            # The main call to the orchestrator
            result = orchestrator.run(prompt, chat_history)

            # Display the results in a clean, structured format
            display_results(result)
            ai_content = result.get("content", result.get("error", ""))
            chat_history.append((prompt, ai_content))

        except KeyboardInterrupt:
            print("\n\nSession interrupted by user. Goodbye!")
            break
        except Exception:
            logging.critical(
                "An unexpected error occurred in the main loop", exc_info=True
            )
            print(
                "\nA critical error occurred. Please check the logs. The session has to end."
            )
            break


if __name__ == "__main__":
    main()
