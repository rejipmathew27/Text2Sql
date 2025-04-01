import streamlit as st
import os
import sqlite3
import pandas as pd
from langchain.agents import create_sql_agent
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.sql_database import SQLDatabase
from langchain.agents import AgentExecutor
from langchain.agents.agent_types import AgentType
from langchain.chat_models import ChatOpenAI
from langchain.callbacks import StreamlitCallbackHandler
import pyreadstat
import requests
import io

# GitHub repository details
GITHUB_REPO = "rejipmathew27/Text2Sql"  # Replace with your GitHub username and repo name
DEFAULT_FILES = ["AE.csv", "DM.csv", "LB.csv", "IE.csv"]

def download_file_from_github(filename):
    """Downloads a file from GitHub."""
    url = f"https://raw.githubusercontent.com/{GITHUB_REPO}/main/{filename}"  # Assuming 'main' branch
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
        return io.StringIO(response.text)  # Return file content as StringIO
    except requests.exceptions.RequestException as e:
        st.error(f"Error downloading {filename}: {e}")
        return None

def read_data(file_like_object, filename):
    """Reads data from CSV, XLSX, or XPT files."""
    try:
        if filename.endswith(".csv"):
            return pd.read_csv(file_like_object)
        elif filename.endswith(".xlsx"):
            return pd.read_excel(file_like_object)
        elif filename.endswith(".xpt"):
            df, meta = pyreadstat.read_xport(file_like_object)
            return df
        else:
            raise ValueError("Unsupported file format.")
    except Exception as e:
        st.error(f"Error reading file: {e}")
        return None

def create_sqlite_db_from_dataframe(df, db_path, table_name):
    """Creates an SQLite database from a Pandas DataFrame."""
    try:
        conn = sqlite3.connect(db_path)
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

    # OpenAI API Key Input
    openai_api_key = st.text_input("Enter your OpenAI API key:", type="password")
    if not openai_api_key:
        st.warning("Please enter your OpenAI API key.")
        return

    # Option to use files from a folder, upload files, or use default files
    source_option = st.radio("Select data source:", ("Upload files", "Use files from folder", "Use default files from GitHub"))

    db_paths = []  # Initialize db_paths outside the conditional blocks

    if source_option == "Upload files":
        uploaded_files = st.file_uploader("Upload CSV, XLSX, or XPT files", type=["csv", "xlsx", "xpt"], accept_multiple_files=True)

        if uploaded_files:
            for uploaded_file in uploaded_files:
                temp_file_path = f"temp_{uploaded_file.name}"
                with open(temp_file_path, "wb") as f:
                    f.write(uploaded_file.getvalue())

                db_path = f"temp_{os.path.splitext(uploaded_file.name)[0]}.db"
                df = read_data(temp_file_path, uploaded_file.name)
                if df is not None:
                    if create_sqlite_db_from_dataframe(df, db_path, os.path.splitext(uploaded_file.name)[0]):
                        db_paths.append(db_path)
                os.remove(temp_file_path)

    elif source_option == "Use files from folder":
        folder_path = st.text_input("Enter the folder path containing CSV, XLSX, or XPT files:")
        if folder_path and os.path.exists(folder_path) and os.path.isdir(folder_path):
            supported_files = [f for f in os.listdir(folder_path) if f.endswith((".csv", ".xlsx", ".xpt"))]
            if supported_files:
                for file in supported_files:
                    file_path = os.path.join(folder_path, file)
                    db_path = f"{os.path.splitext(file)[0]}.db"
                    df = read_data(file_path, file)
                    if df is not None:
                        if create_sqlite_db_from_dataframe(df, db_path, os.path.splitext(file)[0]):
                            db_paths.append(db_path)
            else:
                st.warning("No supported files found in the specified folder.")
        elif folder_path:
            st.error("Invalid folder path.")

    elif source_option == "Use default files from GitHub":
        for filename in DEFAULT_FILES:
            file_like_object = download_file_from_github(filename)
            if file_like_object:
                db_path = f"{os.path.splitext(filename)[0]}.db"
                df = read_data(file_like_object, filename)
                if df is not None:
                    if create_sqlite_db_from_dataframe(df, db_path, os.path.splitext(filename)[0]):
                        db_paths.append(db_path)

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

if __name__ == "__main__":
    main()
