import streamlit as st


def display_debug_sidebar():
    """Renders the debug sidebar with options to clear history and view memory."""
    with st.sidebar:
        st.title("🛠️ Debug Panel")
        if st.button("Clear Chat History"):
            # Clear all session state messages and reset viewer
            st.session_state.messages = []
            st.session_state.show_sql_for_index = None
            st.rerun()

        # Safely access the memory service from session state
        memory_service = st.session_state.get("memory_service")
        if memory_service:
            with st.expander("🧠 Conversation Memory", expanded=False):
                # Use getattr for safer access to internal state
                memory_state = getattr(memory_service, "_state", {})
                st.json(
                    {
                        "summary": memory_state.get("summary", "Not available."),
                        "buffer": memory_state.get("message_buffer", []),
                    }
                )

        with st.expander("📜 Raw Chat History", expanded=False):
            st.json(st.session_state.get("messages", []))
