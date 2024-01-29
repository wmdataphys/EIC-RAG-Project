import os, sys
import streamlit as st

os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
os.environ["TRUBRICS_EMAIL"] = st.secrets["TRUBRICS_EMAIL"]
os.environ["TRUBRICS_PASSWORD"] = st.secrets["TRUBRICS_PASSWORD"]

if st.secrets.get("LANGCHAIN_API_KEY"):
    os.environ["LANGCHAIN_API_KEY"] = st.secrets["LANGCHAIN_API_KEY"]
    os.environ["LANGCHAIN_TRACING_V2"] = st.secrets["LANGCHAIN_TRACING_V2"]
    os.environ["LANGCHAIN_PROJECT"] = st.secrets["LANGCHAIN_PROJECT"]
    os.environ["LANGCHAIN_ENDPOINT"] = st.secrets["LANGCHAIN_ENDPOINT"]
    
st.set_page_config(page_title=, page_icon=, layout="wide")

st.set_page_config(
    page_title="AI4EIC-RAG QA-ChatBot",
    page_icon="https://indico.bnl.gov/event/19560/logo-410523303.png",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://www.extremelycoolapp.com/help',
        'Report a bug': "https://github.com/wmdataphys/EIC-RAG-Project",
        'About': "# This is a header. This is an *extremely* cool app!"
    }
)
st.warning("This project is being continuously developed. Please report any feedback to ai4eic@gmail.com")
col_l, col1, col2, col_r = st.columns([1, 3, 3, 1])

with col1:
    st.image("https://indico.bnl.gov/event/19560/logo-410523303.png")
with col2:
    st.title("""AI4EIC-RAG System""", anchor = "AI4EIC-RAG-QA-Bot", help = "Will Link to arxiv proceeding here.")
    

st.header("About")
st.write(""" 
    This is the AI4EIC-RAG System. It is a QA-ChatBot designed to assist with [specific functionality]. 
    This project is being continuously developed. Please report any feedback to ai4eic@gmail.com.
    """)

st.header("How to Use")
st.write("""
    1. [Step 1]
    2. [Step 2]
    3. [Step 3]
    4. [Step 4]
    """)


st.header("FAQs")
st.write("""
    **Q1: [Question 1]?**
    A1: [Answer 1]

    **Q2: [Question 2]?**
    A2: [Answer 2]
    """)
    
