import streamlit as st
import pandas as pd
import logging
import sys
from pathlib import Path
from dotenv import load_dotenv

# --- PATH SETUP ---
# This is a crucial step to ensure the app can find modules like 'utils' and 'agents'
# We add the project root directory (which is the parent of 'interface/') to the Python path.
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

# --- Core Logic and Utility Imports ---
# Now these imports will work because the project root is in sys.path
from core.utils.config_parser import PROJECT_ROOT, load_app_config
from core.agents import SQLAgent
from core.memory import ConversationMemoryService

from interface.sidebar import display_debug_sidebar
from interface.chat import display_chat_messages

# --- Page Configuration (Must be the first Streamlit command) ---
st.set_page_config(
    page_title="AuraBI - Intelligent Data Assistant", page_icon="✨", layout="wide"
)


# --- Load Custom CSS ---
def load_css(file_path):
    """Loads a CSS file and injects it into the Streamlit app."""
    with open(file_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


# Use a path relative to this script file
ui_dir = Path(__file__).parent
load_css(ui_dir / "style.css")


# --- Logging and Environment Setup ---
load_dotenv(PROJECT_ROOT / ".env")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


# --- Session State and Initialization ---
def initialize_session():
    """Initializes the session state for the application."""
    if "initialized" not in st.session_state:
        try:
            # We can still use PROJECT_ROOT defined in the config parser
            prompts_path = PROJECT_ROOT / "core" / "prompts"
            if not prompts_path.exists():
                raise FileNotFoundError(f"Prompts directory not found: {prompts_path}")

            app_config = load_app_config()

            st.session_state.memory_service = ConversationMemoryService(
                llm_config=app_config.llms,
                prompts_base_path=prompts_path,
                summarizer_llm_key="google-gemini-2.5-lite",
                max_buffer_size=20,
            )
            st.session_state.sql_agent = SQLAgent.from_config(
                agent_key="sql_agent",
                app_config=app_config,
                prompts_base_path=prompts_path,
            )
            st.session_state.messages = []
            st.session_state.show_sql_for_index = None
            st.session_state.initialized = True
            logger.info("Streamlit session initialized successfully.")
        except Exception as e:
            logger.critical(f"Initialization Failed: {e}", exc_info=True)
            st.error(f"Application Initialization Failed: {e}")
            st.stop()


def handle_user_prompt(prompt: str):
    """
    Handles the user's prompt, runs the agent, and updates the chat history.
    """
    # (No changes needed in this function's logic)
    st.session_state.show_sql_for_index = None
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        with st.spinner("Agent is at work..."):
            # ... (rest of the function is identical)
            try:
                memory_service = st.session_state.memory_service
                sql_agent = st.session_state.sql_agent
                memory_service.add_message(role="human", content=prompt)
                working_context = memory_service.get_context_for_agent()
                chat_history = working_context.get("chat_history", [])

                result = sql_agent.run(
                    natural_language_question=prompt, chat_history=chat_history
                )
                df = result.get("dataframe")
                sql_query = result.get("sql_query", "No SQL generated.")

                response_content = "Here are the results for your query."
                if df is not None and df.empty:
                    response_content = (
                        "The query ran successfully but returned no data."
                    )

                assistant_message = {
                    "role": "assistant",
                    "content": response_content,
                    "sql_query": sql_query,
                    "dataframe": df if df is not None else pd.DataFrame(),
                }
                st.session_state.messages.append(assistant_message)

                ai_memory_content = f"Executed SQL. Result: {'Dataframe returned.' if df is not None and not df.empty else 'No data returned.'}"
                memory_service.add_message(role="ai", content=ai_memory_content)

                st.rerun()

            except Exception as e:
                logger.error(f"An error occurred: {e}", exc_info=True)
                error_message = f"I'm sorry, an error occurred: {e}"
                st.error(error_message)
                st.session_state.messages.append(
                    {"role": "assistant", "content": error_message}
                )


# --- Main Application Execution ---
initialize_session()
display_debug_sidebar()  # Uncomment to show the debug panel

# --- Main Page Layout ---
st.title("✨ AuraBI")
st.caption("Your Intelligent Data Assistant")

display_chat_messages()

if prompt := st.chat_input("Ask a question about your data..."):
    handle_user_prompt(prompt)
