import streamlit as st
import pandas as pd
import logging
import sys
from pathlib import Path
from dotenv import load_dotenv
from sql_formatter.core import format_sql

# --- Core Logic Imports ---
from utils.config_parser import PROJECT_ROOT, load_app_config
from agents.sql_agent import SQLAgent
from memory.service import ConversationMemoryService

# --- Page Configuration (Must be the first Streamlit command) ---
st.set_page_config(
    page_title="AuraBI - Intelligent Data Assistant", page_icon="✨", layout="wide"
)

# --- Custom Styling for an Elegant, Professional Dark Theme ---
st.markdown(
    """
    <style>
    @import url('https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css');
    
    /* Core layout and theme */
    .stApp { background-color: #121212; color: #E0E0E0; }
    [data-testid="stSidebar"] { background-color: #1E1E1E; border-right: 1px solid #3A3A3A;}
    .stChatMessage { background-color: #2E2E2E; border: 1px solid #4A4A4A; border-radius: 10px; box-shadow: 0 4px 8px rgba(0,0,0,0.3); }
    .stButton>button {
    /* Base style */
    font-weight: 600;
    border-radius: 8px;
    border: none; /* Remove the default border */
    background-color: #007ACC; /* A strong, elegant blue */
    color: #FFFFFF;
    padding: 0.5rem 1rem;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.2); /* Add depth */
    
    /* Smooth transition for hover effects */
    transition: all 0.3s ease-in-out;
    }

    .stButton>button:hover {
        /* Hover effects */
        background-color: #0099FF; /* Brighter blue on hover */
        box-shadow: 0 6px 10px rgba(0, 0, 0, 0.3); /* Increase shadow for a "lift" effect */
        transform: translateY(-2px); /* Slight lift on hover */
    }

    .stButton>button:active {
        /* Click effect */
        transform: translateY(0px);
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
    }
    
    /* SQL Viewer Styling */
    .sql-viewer-container {
        background-color: #1E1E1E;
        border: 1px solid #4A4A4A;
        border-radius: 10px;
        padding: 1rem;
        box-shadow: 0 4px 8px rgba(0,0,0,0.3);
    }
    </style>
""",
    unsafe_allow_html=True,
)

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
    if "initialized" not in st.session_state:
        try:
            # Correctly define the prompts path as requested
            prompts_path = PROJECT_ROOT / "core" / "prompts"
            if not prompts_path.exists():
                raise FileNotFoundError(
                    f"Prompts directory not found at the required path: {prompts_path}"
                )

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
            st.error(f"Initialization Failed: {e}")
            st.stop()


# --- UI Helper Functions ---
def display_debug_sidebar():
    with st.sidebar:
        st.title("🛠️ Debug Panel")
        if st.button("Clear Chat History"):
            st.session_state.messages = []
            st.session_state.show_sql_for_index = None
            st.rerun()

        memory_service = st.session_state.get("memory_service")
        if memory_service:
            with st.expander("🧠 Conversation Memory", expanded=False):
                memory_state = getattr(memory_service, "_state", {})
                st.json(
                    {
                        "summary": memory_state.get("summary", ""),
                        "buffer": memory_state.get("message_buffer", []),
                    }
                )

        with st.expander("📜 Raw Chat History", expanded=False):
            st.json(st.session_state.get("messages", []))


# --- Main Application ---
initialize_session()
# display_debug_sidebar()

# Main chat container
chat_container = st.container()

with chat_container:
    st.title("✨ AuraBI")
    st.caption("Your Intelligent Data Assistant")

    # Render the entire chat history
    for i, message in enumerate(st.session_state.messages):
        is_sql_viewer_active = st.session_state.show_sql_for_index == i

        # Dynamically create columns only for the message with an active SQL viewer
        if is_sql_viewer_active:
            chat_col, sql_col = st.columns([0.6, 0.4])  # 60% for chat, 40% for SQL
        else:
            chat_col = st.container()

        with chat_col:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

                # --- Render Rich Content (DataFrames, Buttons) ---
                if (
                    "dataframe" in message
                    and message["dataframe"] is not None
                    and not message["dataframe"].empty
                ):
                    df = message["dataframe"]

                    # --- Elegant Button Placement ---
                    # Use columns for right-alignment of the View SQL button
                    _, btn_col, _ = st.columns([0.75, 0.24, 0.01])
                    with btn_col:
                        if "sql_query" in message:
                            if is_sql_viewer_active:
                                if st.button(
                                    "← Close SQL",
                                    key=f"close_sql_{i}",
                                    use_container_width=True,
                                ):
                                    st.session_state.show_sql_for_index = None
                                    st.rerun()
                            else:
                                if st.button(
                                    "View SQL →",
                                    key=f"view_sql_{i}",
                                    use_container_width=True,
                                ):
                                    st.session_state.show_sql_for_index = i
                                    st.rerun()

                    # Display the dataframe
                    st.dataframe(df, use_container_width=True)

                    # Export button at the bottom-left
                    csv = df.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        "💾 Export to CSV",
                        csv,
                        "query_results.csv",
                        "text/csv",
                        key=f"csv_{i}",
                    )

        # If the SQL viewer is active for this message, render it in the right column
        if is_sql_viewer_active:
            with sql_col:
                with st.container():
                    st.info("Generated SQL Query")
                    formatted_sql = format_sql(message["sql_query"])
                    st.code(formatted_sql, language="sql")

    # Anchor for new messages
    st.container().empty()

# --- Handle New User Input ---
if prompt := st.chat_input("Ask a question about your data..."):
    st.session_state.show_sql_for_index = (
        None  # Close any open SQL viewer on new prompt
    )
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Agent is at work..."):
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
