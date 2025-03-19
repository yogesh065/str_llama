import streamlit as st
import pandas as pd
from langchain_community.agent_toolkits import SQLDatabaseToolkit, create_sql_agent
from langchain_community.utilities import SQLDatabase
from langchain_groq import ChatGroq
from sqlalchemy import create_engine, StaticPool, inspect
import sqlite3
import time
import re
import datetime
import re
# App layout
st.set_page_config(page_title="AI Data scientist", layout="wide", page_icon="🤖")
st.title("📈 Smart Data Analysis Chat with Yogesh")
# Configuration constants
DB_CONFIG = {
    "url": "sqlite:///:memory:?cache=shared",
    "poolclass": StaticPool,
    "creator": lambda: sqlite3.connect("file::memory:?cache=shared", uri=True)
}
MODEL_CONFIG = {
    "model": "llama-3.3-70b-specdec",
    "temperature": 0.7,
    "max_tokens":1000,
}

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "engine" not in st.session_state:
    st.session_state.engine = None

# Custom CSS styling
st.markdown("""
<style>
    .stChatInput {position: fixed; bottom: 20px; width: 70%;}
    .stChatMessage {
        border-radius: 15px !important;
        padding: 1.2rem !important;
        margin: 0.8rem 0 !important;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1) !important;
    }
    [data-testid="stSidebar"] {
        background: #f0f2f6 !important;
        padding: 20px !important;
    }
    .header-text { font-size: 1.4rem !important; }
</style>
""", unsafe_allow_html=True)

def initialize_llm():
    """Initialize Groq language model"""
    return ChatGroq(
        api_key=st.secrets["key_api"]["GROQ_API_KEY"],
        **MODEL_CONFIG
    )



def excel_to_sqlite(excel_file):
    """Process Excel file to SQLite database with clean state"""
    try:
        engine = create_engine(**DB_CONFIG)
        
        # Clear existing tables before processing new file
        with engine.begin() as conn:
            inspector = inspect(engine)
            existing_tables = inspector.get_table_names()
            for table in existing_tables:
                # Use double quotes around table names to handle spaces
                conn.execute(f"DROP TABLE IF EXISTS \"{table}\"")
                
        # Rest of the processing remains the same
        with pd.ExcelFile(excel_file) as excel_data:
            for sheet_name in excel_data.sheet_names:
                # Clean sheet name to ensure it's a valid SQL identifier
                cleaned_sheet_name = re.sub(r'\W+', '_', sheet_name)
                
                df = pd.read_excel(excel_data, sheet_name=sheet_name)
                df.columns = [re.sub(r'\W+', '_', str(col)) for col in df.columns]
                
                for col in df.columns:
                    if df[col].dtype == 'datetime64[ns]' or df[col].dtype == 'datetime64[ns, UTC]':
                        df[col] = df[col].astype(str)
                    elif df[col].dtype == 'object':
                        df[col] = df[col].apply(lambda x: str(x) if isinstance(x, (datetime.time, datetime.date)) else x)
                
                if not df.empty and len(df.columns) > 0:
                    df.to_sql(
                        name=cleaned_sheet_name,
                        con=engine,
                        if_exists='replace',
                        index=False
                    )
        return engine, None
    except Exception as e:
        return None, str(e)



def create_agent(engine):
    """Create SQL analysis agent"""
    db = SQLDatabase(engine)
    toolkit = SQLDatabaseToolkit(db=db, llm=initialize_llm())
    return create_sql_agent(
        llm=initialize_llm(),
        toolkit=toolkit,
        agent_type="openai-tools",
        verbose=True
    )



# Sidebar controls
with st.sidebar:
    st.header("Data Controls")
    uploaded_file = st.file_uploader("Upload Excel File", type=["xlsx", "xls"])
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.success("History cleared!")
    st.markdown("---")
    st.caption("Last 5 conversations will be preserved")

# Main chat interface
chat_container = st.container()

# File processing
if uploaded_file and st.session_state.engine is None:
    with st.status("Processing your data...", expanded=True) as status:
        engine, error = excel_to_sqlite(uploaded_file)
        if engine:
            st.session_state.engine = engine
            inspector = inspect(engine)
            tables = inspector.get_table_names()
            status.update(label=f"✅ Loaded {len(tables)} tables: {', '.join(tables)}", state="complete")
        else:
            st.error(f"Error processing file: {error}")

# Display chat messages
with chat_container:
    for message in st.session_state.messages[-5:]:  # Show last 5 messages
        with st.chat_message(message["role"], avatar="💼" if message["role"] == "user" else "🤖"):
            st.markdown(message["content"])

# Handle user input
if prompt := st.chat_input("Ask about your data..."):
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with chat_container:
        with st.chat_message("user", avatar="💼"):
            st.markdown(prompt)
    
    # Process query
    if st.session_state.engine:
        with st.spinner("Analyzing data..."):
            try:
                agent = create_agent(st.session_state.engine)
                start_time = time.time()
                result = agent.invoke({"input": prompt})
                response = f"{result['output']}\n\n*Analysis took {time.time()-start_time:.1f} seconds*"
                
                # Add AI response
                st.session_state.messages.append({"role": "assistant", "content": response})
                
                # Display response
                with chat_container:
                    with st.chat_message("assistant", avatar="🤖"):
                        st.markdown(response)
                        
            except Exception as e:
                error_msg = f"❌ Analysis error: {str(e)}"
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
                with chat_container:
                    with st.chat_message("assistant", avatar="⚠️"):
                        st.markdown(error_msg)
    else:
        with chat_container:
            with st.chat_message("assistant", avatar="⚠️"):
                warning = "Please upload an Excel file first!"
                st.markdown(warning)
                st.session_state.messages.append({"role": "assistant", "content": warning})

    # Maintain max 5 messages
    if len(st.session_state.messages) > 5:
        st.session_state.messages = st.session_state.messages[-5:]
