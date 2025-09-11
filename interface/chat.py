import streamlit as st
from sql_formatter.core import format_sql


def display_chat_messages():
    """
    Renders the chat history, including dataframes, SQL viewers,
    and action buttons.
    """
    for i, message in enumerate(st.session_state.get("messages", [])):
        is_sql_viewer_active = st.session_state.get("show_sql_for_index") == i

        # Use columns for the SQL viewer layout
        chat_col, sql_col = (
            st.columns([0.6, 0.4]) if is_sql_viewer_active else (st.container(), None)
        )

        with chat_col:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

                # Render dataframe if it exists and is not empty
                if "dataframe" in message and not message["dataframe"].empty:
                    render_dataframe_and_controls(message, i, is_sql_viewer_active)

        # Render the SQL viewer in the second column if active
        if is_sql_viewer_active and sql_col:
            with sql_col:
                with st.container(
                    border=True
                ):  # Using border for better visual separation
                    st.info("Generated SQL Query")
                    formatted_sql = format_sql(message["sql_query"])
                    st.code(formatted_sql, language="sql")


def render_dataframe_and_controls(message, index, is_sql_viewer_active):
    """Renders the dataframe and its associated buttons (View/Close SQL, Export)."""
    df = message["dataframe"]

    # Button placement using columns for alignment
    _, btn_col = st.columns([0.75, 0.25])  # Align button to the right
    with btn_col:
        if "sql_query" in message:
            button_label = "← Close SQL" if is_sql_viewer_active else "View SQL →"
            if st.button(
                button_label, key=f"sql_toggle_{index}", use_container_width=True
            ):
                st.session_state.show_sql_for_index = (
                    None if is_sql_viewer_active else index
                )
                st.rerun()

    # Display the dataframe
    st.dataframe(df, use_container_width=True)

    # Export button
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "💾 Export to CSV",
        csv,
        "query_results.csv",
        "text/csv",
        key=f"csv_{index}",
    )
