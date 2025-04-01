import streamlit as st
import os
import sqlite3
import pandas as pd
from langchain.agents import create_sql_agent
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.sql_database import SQLDatabase
from langchain.llms import OpenAI
from langchain.agents import AgentExecutor
from langchain.agents.agent_types import AgentType
from langchain.chat_models import ChatOpenAI
from langchain.callbacks import StreamlitCallbackHandler

# Set OpenAI API key
openai_api_key = st.secrets["OPENAI_API_KEY"]

def create_sqlite_db_from_csv(csv_file_path, db_path):
    """Creates an SQLite database from a CSV file."""
    try:
        df = pd.read_csv(csv_file_path)
        conn = sqlite3.connect(db_path)
        table_name = os.path.splitext(os.path.basename(csv_file_path))[0]
        df.to_sql(table_name, conn, if_exists="replace", index=False)
        conn.close()
        return True
    except Exception as e:
        st.error(f"Error creating database: {e}")
        return False

def create_sql_agent_from_db(db_path, llm):
    """Creates a SQL agent from an SQLite database."""
    try:
        db = SQLDatabase.from_uri(f"sqlite:///{db_path}")
        toolkit = SQLDatabaseToolkit(db=db, llm=llm)
        agent_executor = create_sql_agent(
            llm=llm,
            toolkit=toolkit,
            verbose=True,
            agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        )
        return agent_executor
    except Exception as e:
        st.error(f"Error creating agent: {e}")
        return None

def main():
    st.title("Text to SQL Agent")

    # Option to use files from a folder or upload files
    source_option = st.radio("Select data source:", ("Upload files", "Use files from folder"))

    if source_option == "Upload files":
        uploaded_files = st.file_uploader("Upload CSV files", type=["csv"], accept_multiple_files=True)

        if uploaded_files:
            db_paths = []
            for uploaded_file in uploaded_files:
                temp_csv_path = f"temp_{uploaded_file.name}"
                with open(temp_csv_path, "wb") as f:
                    f.write(uploaded_file.getvalue())

                db_path = f"temp_{os.path.splitext(uploaded_file.name)[0]}.db"
                if create_sqlite_db_from_csv(temp_csv_path, db_path):
                    db_paths.append(db_path)
                os.remove(temp_csv_path)

            if db_paths:
                query = st.text_input("Enter your SQL query in natural language:")
                if query:
                    llm = ChatOpenAI(temperature=0, verbose=True, openai_api_key=openai_api_key, streaming=True)
                    for db_path in db_paths:
                        agent_executor = create_sql_agent_from_db(db_path, llm)
                        if agent_executor:
                            try:
                                st_cb = StreamlitCallbackHandler(st.container())
                                response = agent_executor.run(query, callbacks=[st_cb])
                                st.write(response)

                            except Exception as e:
                                st.error(f"Error executing query: {e}")
                        else:
                            st.error("Agent creation failed.")
                    for db_path in db_paths:
                        os.remove(db_path)

    elif source_option == "Use files from folder":
        folder_path = st.text_input("Enter the folder path containing CSV files:")

        if folder_path and os.path.exists(folder_path) and os.path.isdir(folder_path):
            csv_files = [f for f in os.listdir(folder_path) if f.endswith(".csv")]

            if csv_files:
                db_paths = []
                for csv_file in csv_files:
                    csv_file_path = os.path.join(folder_path, csv_file)
                    db_path = f"{os.path.splitext(csv_file)[0]}.db"
                    if create_sqlite_db_from_csv(csv_file_path, db_path):
                        db_paths.append(db_path)

                query = st.text_input("Enter your SQL query in natural language:")
                if query:
                    llm = ChatOpenAI(temperature=0, verbose=True, openai_api_key=openai_api_key, streaming=True)
                    for db_path in db_paths:
                        agent_executor = create_sql_agent_from_db(db_path, llm)
                        if agent_executor:
                            try:
                                st_cb = StreamlitCallbackHandler(st.container())
                                response = agent_executor.run(query, callbacks=[st_cb])
                                st.write(response)

                            except Exception as e:
                                st.error(f"Error executing query: {e}")
                        else:
                            st.error("Agent creation failed.")
                    for db_path in db_paths:
                        os.remove(db_path)

            else:
                st.warning("No CSV files found in the specified folder.")
        elif folder_path:
            st.error("Invalid folder path.")

if __name__ == "__main__":
    main()
