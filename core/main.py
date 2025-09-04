import os
import logging
import sys
from pathlib import Path
from core.utils.config_parser import load_app_config
from core.agents.sql_agent import SQLAgent
from core.memory.service import ConversationMemoryService
from dotenv import load_dotenv


load_dotenv("/home/muhammad/machine_learning/AI_Agents/bi_agent/.env")

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


# --- Main Application Logic ---
def run_conversation():
    """
    Runs a simple, interactive conversational loop to demonstrate the
    new threshold-based memory system.
    """
    logger.info("Initializing conversational AI system...")

    # --- 1. Load Configuration ---
    app_config = load_app_config()
    prompts_base_path = Path("core/prompts")

    # --- 2. Instantiate Services and Agents ---
    # Instantiate the new memory service with its configuration
    memory_service = ConversationMemoryService(
        llm_config=app_config.llms,
        prompts_base_path=prompts_base_path,
        summarizer_llm_key="google-gemini-2.5-lite",  # Use a fast, cheap model for summaries
        max_buffer_size=6,  # e.g., 3 pairs of Q&A before summarizing
    )

    # The worker agent instantiation remains the same
    sql_agent = SQLAgent.from_config(
        agent_key="sql_agent",
        app_config=app_config,
        prompts_base_path=prompts_base_path,
    )
    logger.info("System initialized. You can start the conversation.")
    print("\n--- Enter 'exit' to end the session ---")

    # --- 3. Simplified Conversational Loop ---
    while True:
        try:
            user_input = input("Human: ")
            if user_input.lower() == "exit":
                print("AI: Goodbye!")
                break

            # --- Step 1: Add user message to memory ---
            memory_service.add_message(role="human", content=user_input)

            # --- Step 2: Get the current working context for the agent ---
            working_context = memory_service.get_context_for_agent()

            # --- Step 3: Route to a worker agent (hardcoded to SQLAgent for now) ---
            # The worker agent's prompt should be updated to accept the summary.
            # For now, we pass the chat_history which is the buffer.
            logger.info("Routing to SQLAgent...")
            try:
                # The agent receives the latest buffer of conversation
                result_df = sql_agent.run(
                    natural_language_question=user_input,
                    chat_history=working_context["chat_history"],
                )
                agent_output = f"""
                Successfully executed the query:
                ```sql
                {result_df["sql_query"]}
                ``` 
                Retrieved {len(result_df["dataframe"])} rows."
                """

                print(f"AI: {agent_output}")
                print(result_df["dataframe"].to_markdown(index=False))

            except RuntimeError as e:
                agent_output = f"I encountered an error: {e}"
                print(f"AI: {agent_output}")

            # --- Step 4: Add the AI's response to memory ---
            # This might trigger summarization if the buffer is now full.
            memory_service.add_message(role="ai", content=agent_output)

        except (KeyboardInterrupt, EOFError):
            print("\nAI: Conversation ended.")
            break
        except Exception as e:
            logger.critical(
                f"An unexpected error occurred in the conversation loop: {e}",
                exc_info=True,
            )
            print("AI: I'm sorry, a critical error occurred. The session has to end.")
            break


if __name__ == "__main__":
    run_conversation()
