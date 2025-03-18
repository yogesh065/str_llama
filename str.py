import streamlit as st
import pandas as pd
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.utilities import SQLDatabase
from langchain import create_sql_agent
import sqlite3
import os
from groq import Groq

# Set up the Streamlit app
st.set_page_config(page_title="Excel to SQL Agent", layout='wide')

# Set up the Groq API key
api_key = st.secrets["key_api"]["GROQ_API_KEY"]

# Function to convert Excel to SQLite
def excel_to_sqlite(excel_file):
    # Create a SQLite database
    db_name = "excel_data.db"
    if os.path.exists(db_name):
        os.remove(db_name)
    conn = sqlite3.connect(db_name)
    
    # Convert each sheet to a table
    for sheet_name in excel_file.sheet_names:
        df = pd.read_excel(excel_file, sheet_name=sheet_name)
        df.to_sql(sheet_name, conn, if_exists='replace', index=False)
    
    conn.close()
    
    return db_name

# Upload Excel file
uploaded_file = st.file_uploader("Choose an Excel file", type=['xlsx'])

if uploaded_file is not None:
    # Convert Excel to SQLite
    db_name = excel_to_sqlite(uploaded_file)
    
    # Create a SQLDatabase object
    db_engine = sqlite3.connect(db_name)
    db = SQLDatabase(db_engine)
    
    # Initialize the Groq model
    groq_model = Groq(api_key=api_key)
    
    # Create a custom LLM wrapper for Groq
    class GroqLLM:
        def __init__(self, model):
            self.model = model
        
        async def __call__(self, prompt):
            messages = [{"role": "user", "content": prompt}]
            completion = self.model.chat.completions.create(
                model="llama-3.3-70b-specdec",
                messages=messages,
                temperature=1,
                max_tokens=8000,
                top_p=1,
                stream=True,
                stop=None,
            )
            full_response = ''
            for chunk in completion:
                if chunk.choices[0].delta.content:
                    for ch in chunk.choices[0].delta.content:
                        full_response += ch
            return full_response
    
    # Create a SQL agent using the Groq LLM
    llm = GroqLLM(groq_model)
    toolkit = SQLDatabaseToolkit(db=db, llm=llm)
    agent_executor = create_sql_agent(llm=llm, db=db, agent_type="openai-tools", verbose=True)
    
    # User input for query
    query = st.text_input("Enter your query (e.g., 'What is the average value in column X of sheet Y?')")
    
    if query:
        try:
            # Invoke the SQL agent
            response = agent_executor.invoke({"input": query})
            st.write("Response:", response['output'])
        except Exception as e:
            st.error("Error:", str(e))

