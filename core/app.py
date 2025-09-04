import chainlit as cl
import logging
import sys
from pathlib import Path
from dotenv import load_dotenv
import json 

from core.utils.config_parser import PROJECT_ROOT, load_app_config
from core.agents.sql_agent import SQLAgent
from core.memory.service import ConversationMemoryService


load_dotenv(PROJECT_ROOT / ".env")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


app_config = None
prompts_base_path = None
sql_agent = None
memory_service = None

MEMORY_LOG_FILE_PATH = Path(__file__).parent.parent / "conversation_memory_log.json"


def log_memory_state(memory_service: ConversationMemoryService, step: str):
    """
    Logs the current state of the ConversationMemoryService to a file.
    This function safely accesses the internal _state of the service.
    """
    global MEMORY_LOG_FILE_PATH
    try:
        state_to_log = {
            "step": step,
            "summary": memory_service._state.get("summary", ""),
            "message_buffer": memory_service._state.get("message_buffer", []),
        }

        log_data = []
        if MEMORY_LOG_FILE_PATH.exists():
            try:
                with open(MEMORY_LOG_FILE_PATH, "r", encoding="utf-8") as f:
                    log_data = json.load(f)
                    if not isinstance(log_data, list):
                        log_data = []  # Ensure it's a list
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(
                    f"Could not read existing memory log file {MEMORY_LOG_FILE_PATH}: {e}. Starting fresh."
                )

        log_data.append(state_to_log)

        with open(MEMORY_LOG_FILE_PATH, "w", encoding="utf-8") as f:
            json.dump(log_data, f, indent=4, ensure_ascii=False)

        logger.info(f"Memory state logged to {MEMORY_LOG_FILE_PATH} at step: {step}")

    except Exception as e:
        logger.error(f"Failed to log memory state: {e}")


@cl.on_chat_start
async def start():
    """Initialize the application when the chat starts."""
    global app_config, prompts_base_path, sql_agent, memory_service
    try:
        app_config = load_app_config()
        prompts_base_path = PROJECT_ROOT / "prompts"

        memory_service = ConversationMemoryService(
            llm_config=app_config.llms,
            prompts_base_path=prompts_base_path,
            summarizer_llm_key="google-gemini-2.5-lite",
            max_buffer_size=20,  # Example: 10 pairs of Q&A before summarizing
        )

        sql_agent = SQLAgent.from_config(
            agent_key="sql_agent",
            app_config=app_config,
            prompts_base_path=prompts_base_path,
        )

        log_memory_state(memory_service, "on_chat_start_initialized")

        logger.info(
            "Chainlit application initialized successfully with SQL Agent and Memory Service."
        )
        await cl.Message(
            content="Hello! I'm your SQL Agent. Ask me questions about your database. I'll remember our conversation. (Memory logging enabled)"
        ).send()

    except Exception as e:
        logger.error(f"Failed to initialize application: {e}", exc_info=True)
        await cl.Message(
            content="Sorry, I encountered an error while initializing. Please check the application logs."
        ).send()


@cl.on_message
async def main(message: cl.Message):
    """Handle incoming messages from the user."""
    global sql_agent, memory_service

    if sql_agent is None or memory_service is None:
        await cl.Message(
            content="Sorry, the application components are not initialized correctly. Please restart the chat."
        ).send()
        logger.error("SQL Agent or Memory Service is not initialized.")
        return

    # Get the user's question
    user_question = message.content
    logger.info(f"Received user question: {user_question}")

    # --- Log Memory State BEFORE adding user message ---
    log_memory_state(memory_service, f"before_user_message: {user_question[:30]}...")

    # Add user message to memory
    memory_service.add_message(role="human", content=user_question)

    # --- Log Memory State AFTER adding user message ---
    log_memory_state(
        memory_service, f"after_user_message_added: {user_question[:30]}..."
    )

    # Get the current working context (summary + recent history) for the agent
    working_context = memory_service.get_context_for_agent()
    chat_history_for_agent = working_context.get("chat_history", [])

    # --- Log the Working Context Sent to Agent ---
    try:
        context_log_path = MEMORY_LOG_FILE_PATH.with_name("agent_context_log.json")
        context_entry = {
            "user_question": user_question,
            "working_context_sent": working_context,
        }
        context_log_data = []
        if context_log_path.exists():
            try:
                with open(context_log_path, "r", encoding="utf-8") as f:
                    context_log_data = json.load(f)
                    if not isinstance(context_log_data, list):
                        context_log_data = []
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(
                    f"Could not read existing context log file {context_log_path}: {e}. Starting fresh."
                )

        context_log_data.append(context_entry)

        with open(context_log_path, "w", encoding="utf-8") as f:
            json.dump(context_log_data, f, indent=4, ensure_ascii=False)
        logger.info(f"Agent context logged to {context_log_path}")
    except Exception as e:
        logger.error(f"Failed to log agent context: {e}")

    # Create a temporary message to show the user that we're processing
    processing_msg = cl.Message(
        content="🔍 Analyzing your question and querying the database..."
    )
    await processing_msg.send()

    try:
        # --- Run the SQL agent with conversation history ---
        logger.info("Routing query to SQLAgent...")
        result = sql_agent.run(
            natural_language_question=user_question,
            chat_history=chat_history_for_agent,  # Pass the history
        )

        # Extract the results from the agent's response
        df = result["dataframe"]
        generated_sql = result["sql_query"]

        # Prepare elements to send (e.g., the DataFrame)
        elements = []
        if df is not None and not df.empty:
            elements.append(
                cl.Dataframe(data=df, display="inline", name="Query Results")
            )
        elif df is not None and df.empty:
            pass  # Message content will indicate this

        # Prepare the response message content
        response_content = f"✅ I've executed the following query for you:\n\n```sql\n{generated_sql}\n```"
        if df is not None and df.empty:
            response_content += (
                "\n\nℹ️ The query executed successfully, but returned no rows."
            )

        # --- Log Memory State BEFORE adding AI message ---
        log_memory_state(
            memory_service, f"before_ai_message_added_for: {user_question[:30]}..."
        )

        # Add AI response to memory
        ai_memory_content = (
            f"Successfully executed query: ```sql\n{generated_sql}\n```\n"
        )
        if df is not None:
            rows_returned = len(df)
            ai_memory_content += (
                f"Retrieved {rows_returned} row{'s' if rows_returned != 1 else ''}."
            )
            # Optional: Add brief data summary if needed

        memory_service.add_message(role="ai", content=ai_memory_content)
        logger.info(f"Added AI response to memory: {ai_memory_content}")

        # --- Log Memory State AFTER adding AI message ---
        log_memory_state(
            memory_service, f"after_ai_message_added_for: {user_question[:30]}..."
        )

        # Update the processing message with the final results
        processing_msg.content = response_content
        processing_msg.author = "SQL Agent"
        processing_msg.elements = elements
        await processing_msg.update()

    except RuntimeError as e:
        error_message = f"❌ I encountered an error while processing your query: {e}"
        logger.error(f"SQL Agent runtime error: {e}")

        # --- Log Memory State BEFORE adding AI error message ---
        log_memory_state(
            memory_service,
            f"before_ai_error_message_added_for: {user_question[:30]}...",
        )

        # Add error to memory
        memory_service.add_message(role="ai", content=f"Error occurred: {e}")

        # --- Log Memory State AFTER adding AI error message ---
        log_memory_state(
            memory_service, f"after_ai_error_message_added_for: {user_question[:30]}..."
        )

        # Update the processing message for error
        processing_msg.content = error_message
        processing_msg.author = "SQL Agent"
        await processing_msg.update()

    except Exception as e:
        error_message = f"❌ An unexpected error occurred: {e}"
        logger.error(f"Unexpected error in Chainlit app: {e}", exc_info=True)

        # --- Log Memory State BEFORE adding unexpected error message ---
        log_memory_state(
            memory_service,
            f"before_unexpected_error_message_added_for: {user_question[:30]}...",
        )

        # Add error to memory
        memory_service.add_message(role="ai", content=f"Unexpected error occurred: {e}")

        # --- Log Memory State AFTER adding unexpected error message ---
        log_memory_state(
            memory_service,
            f"after_unexpected_error_message_added_for: {user_question[:30]}...",
        )

        # Update the processing message for unexpected error
        processing_msg.content = error_message
        processing_msg.author = "System"
        await processing_msg.update()


@cl.on_chat_start
async def start():
    """Initialize the application when the chat starts."""
    global app_config, prompts_base_path, sql_agent, memory_service
    try:
        # 1. Load configuration
        app_config = load_app_config()
        prompts_base_path = Path("prompts")  # Adjusted path relative to project root

        # 2. Initialize the Conversation Memory Service
        # Use the same LLM config and prompts path
        # Configure summarizer LLM key and buffer size as needed
        memory_service = ConversationMemoryService(
            llm_config=app_config.llms,
            prompts_base_path=prompts_base_path,
            summarizer_llm_key="google-gemini-2.5-lite",  # Or load from config if preferred
            max_buffer_size=6,  # Example: 3 pairs of Q&A before summarizing
        )

        # 3. Initialize the SQL agent using the factory method
        sql_agent = SQLAgent.from_config(
            agent_key="sql_agent",
            app_config=app_config,
            prompts_base_path=prompts_base_path,
        )

        logger.info(
            "Chainlit application initialized successfully with SQL Agent and Memory Service."
        )
        await cl.Message(
            content="Hello! I'm your SQL Agent. Ask me questions about your database. I'll remember our conversation."
        ).send()

    except Exception as e:
        logger.error(f"Failed to initialize application: {e}", exc_info=True)
        await cl.Message(
            content="Sorry, I encountered an error while initializing. Please check the application logs."
        ).send()


@cl.on_message
async def main(message: cl.Message):
    """Handle incoming messages from the user."""
    global sql_agent, memory_service

    # Safety check for agent initialization
    if sql_agent is None or memory_service is None:
        await cl.Message(
            content="Sorry, the application components are not initialized correctly. Please restart the chat."
        ).send()
        logger.error("SQL Agent or Memory Service is not initialized.")
        return

    # Get the user's question
    user_question = message.content
    logger.info(f"Received user question: {user_question}")

    # Add user message to memory
    memory_service.add_message(role="human", content=user_question)

    # Get the current working context (summary + recent history) for the agent
    working_context = memory_service.get_context_for_agent()
    chat_history_for_agent = working_context.get("chat_history", [])

    # Create a temporary message to show the user that we're processing
    processing_msg = cl.Message(
        content="🔍 Analyzing your question and querying the database..."
    )
    await processing_msg.send()

    try:
        # --- Run the SQL agent with conversation history ---
        logger.info("Routing query to SQLAgent...")
        result = sql_agent.run(
            natural_language_question=user_question,
            chat_history=chat_history_for_agent,  # Pass the history
        )

        # Extract the results from the agent's response
        df = result["dataframe"]
        generated_sql = result["sql_query"]

        # Prepare elements to send (e.g., the DataFrame)
        elements = []
        if df is not None and not df.empty:
            elements.append(
                cl.Dataframe(data=df, display="inline", name="Query Results")
            )
        elif df is not None and df.empty:
            # Handle case where query was successful but returned no rows
            pass  # Message content will indicate this
        # If df is None, an error must have occurred, handled below or in agent's run

        # Prepare the response message content
        response_content = f"✅ I've executed the following query for you:\n\n```sql\n{generated_sql}\n```"
        if df is not None and df.empty:
            response_content += (
                "\n\nℹ️ The query executed successfully, but returned no rows."
            )

        # Add AI response to memory *before* sending to user
        # Include the SQL and maybe a snippet of the data in the memory if needed,
        # but the full DF isn't necessary for memory.
        ai_memory_content = (
            f"Successfully executed query: ```sql\n{generated_sql}\n```\n"
        )
        if df is not None:
            rows_returned = len(df)
            ai_memory_content += (
                f"Retrieved {rows_returned} row{'s' if rows_returned != 1 else ''}."
            )
            if rows_returned > 0:
                # Add a brief textual summary of the data if useful for context
                # This is optional and depends on data nature
                pass  # For now, just log the fact data was returned

        memory_service.add_message(role="ai", content=ai_memory_content)
        logger.info(f"Added AI response to memory: {ai_memory_content}")

        # Update the processing message with the final results
        # Correct way: Update attributes first, then call update()
        processing_msg.content = response_content
        processing_msg.author = "SQL Agent"
        processing_msg.elements = elements
        await processing_msg.update()

    except RuntimeError as e:
        # Handle errors originating from the SQL agent's internal logic or execution
        error_message = f"❌ I encountered an error while processing your query: {e}"
        logger.error(f"SQL Agent runtime error: {e}")

        # Add error to memory so the context is preserved
        memory_service.add_message(role="ai", content=f"Error occurred: {e}")

        # Update the processing message for error
        processing_msg.content = error_message
        processing_msg.author = "SQL Agent"
        await processing_msg.update()

    except Exception as e:
        # Handle unexpected errors in the Chainlit UI logic
        error_message = f"❌ An unexpected error occurred: {e}"
        logger.error(f"Unexpected error in Chainlit app: {e}", exc_info=True)

        # Add error to memory
        memory_service.add_message(role="ai", content=f"Unexpected error occurred: {e}")

        # Update the processing message for unexpected error
        processing_msg.content = error_message
        processing_msg.author = "System"
        await processing_msg.update()


# --- Future Agent Integration (Conceptual) ---
# To add more agents, you could modify the logic here.
# For example:
# 1. Determine the agent based on user input or settings (e.g., keyword, dropdown)
#    agent_type = determine_agent_type(user_question) # Implement this logic
# 2. Dispatch to the appropriate agent instance
#    if agent_type == "sql":
#        result = sql_agent.run(...)
#    elif agent_type == "plot":
#        result = plotting_agent.run(...) # Assuming you create this
# 3. Handle the result accordingly, potentially sending different types of elements (e.g., cl.Plotly)
