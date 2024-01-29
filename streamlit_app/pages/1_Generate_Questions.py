from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader

from langchain.schema import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough

from langchain.prompts import PromptTemplate

import streamlit as st
import numpy as np
import arxiv, os
from operator import itemgetter

from app_utilities import num_tokens_from_prompt

os.environ["LANGCHAIN_PROJECT"] = st.secrets["LANGCHAIN_EVAL_PROJECT"]

st.set_page_config(page_title="AI4EIC-RAG QA-ChatBot", page_icon="https://indico.bnl.gov/event/19560/logo-410523303.png", layout="wide")
st.warning("This project is being continuously developed. Please report any feedback to ai4eic@gmail.com")
col_l, col1, col2, col_r = st.columns([1, 3, 3, 1])
with col1:
    st.image("https://indico.bnl.gov/event/19560/logo-410523303.png")
with col2:
    st.title("""Generate Questions for Evaluations""", anchor = "GenerateQuestions", help = "Will Link to arxiv proceeding here.")

articles = open("Resources/ARXIV_SOURCES.info", "r").readlines()  

def load_article(key: bool):
      st.session_state["QuestionGen"] = key

if not st.session_state.get("Info_Container"):
    st.session_state["Info_Container"] = st.container()
with st.container():
    pressed = st.button("Load an article", on_click = load_article, args = (True,))
    if pressed:
        with st.status("Loading article..."):
            article = np.random.choice(articles).strip("\n")
            st.write(f"Downloading {article} ID from arxiv.org...")
            search = arxiv.Search(id_list=[article])
            paper = next(arxiv.Client().results(search))
        st.session_state["article"] = article
        st.header("Loaded Article ID : " + article, divider = "rainbow")
        st.subheader("Title")
        st.write(paper.title)
        st.subheader("Authors")
        st.write('\n'.join([f'{i+1}. {auth.name}' for i, auth in enumerate(paper.authors)]))
        st.subheader("Categories")
        st.write('\n'.join([f'{i+1}. {cat}' for i, cat in enumerate(paper.categories)]))
        st.subheader("Primary Category")
        st.write(paper.primary_category)
        st.subheader("Published")
        st.write(str(paper.published))
        st.subheader("Abstract")
        st.write(paper.summary)
        st.subheader("Published in Peer reviewed journal")
        st.write(paper.doi)
        st.subheader("PDF link")
        st.write(paper.pdf_url)
        st.session_state["pdf_url"] = paper.pdf_url
        with st.status("Loading PDF into PyPDFLoader...."):
            docs = PyPDFLoader(paper.pdf_url).load()
            st.write("Counting tokens ......")
            content = "\n".join([doc.page_content for doc in docs])
            st.session_state["content"] = content
            num_tokens = num_tokens_from_prompt(content, "gpt-3.5-turbo-1106")
        st.subheader("Number of Pages:" + str(len(docs)))
        st.subheader("Number of Tokens:" + str(num_tokens))
        if (num_tokens > 12000):
            st.warning("This article is too long. Please reload to select a shorter article....")
            st.stop()
    
if not st.session_state.get("QuestionGen"):
    st.stop()

GPTDict = {"3.5": "gpt-3.5-turbo-1106", "4": "gpt-4-0125-preview"}

response = """
{prefix}
Now, Given the text within the tags <content> and </content> Generate {NQUESTIONS} targetted question that contains exactly {NCLAIMS} claims along with its answer. 
Make sure the clearly label your question and answer exactly to the point. 
Also, Make sure to use the exact information from this document and do not use any other information.

THe answer is formatted as a json object and written in `python` format

<content>
{CONTEXT}
</content>
"""
prefix = """
            You are an expert in framing questions from a given content. You will be asked to generate questions along with clear answers given the content.
            Here is an explanation to help you better understand the task:

            A targetted question with `N` claims is defined as a question that contains exactly N claims to answer. 
            For instance, an example of a targetted question with 2 claims is as follows:
            
            Q: When and Where will the Electron Ion Collider be constructed?
            A:  ```python \n
                {"n_claims" : 2, 
                "claims": ["When will Electron Ion Collider be constructed", "Where will Electron Ion Collider be constructed"], 
                "complete_response": " The Electron Ion Collider will be constructed in Long Island in NewYork by the end of 2035. \n", 
                "answers": ["Long Island in NewYork", "2035"], 
                "relevance_score": 100
                } \n
                ```
            
            Another example of a question with 3 claims is shown below
            
            Q: What is the dimension of the MAPS pixel layer in ITS3 EIC techonology? How many layers of MAPS detector will be in EIC? What is the thickness of the MAPS layer?
            A:  ```python
                {"n_claims" : 3, 
                "claims": ["dimension of MAPS pixel layer", "number of layers of MAPS detector", "thickness of MAPS layer"], 
                "complete_response": " Dimensions of MAPS pixel layer is 10x10 mm. \n There are a total of 7 layers in MAPS detector at EIC. \n The thickness of each layer of MAPS detector is 5um. ", 
                "answers": ["10x10mm", "7", "5um"], 
                "relevance_score": 100
                }
                ```
            """
prompt = PromptTemplate.from_template(response)
def gen_submit(key: bool):
    st.session_state["Generate"] = key
with st.container():
    with st.form("Generate Question"):
        n_claims = st.number_input("Number of claims to be generated in each question", min_value = 1, max_value = 10)
        n_questions = st.number_input("Number of questions to be generated", min_value = 1, max_value = 5)
        n_iterations = st.number_input("Number of iterations for each question", min_value = 1, max_value = 5)
        GPTVersion = st.selectbox("GPT Version", ["4"])
        st.form_submit_button("Generate", on_click = gen_submit, args = (True,))
        if st.session_state.get("Generate"):
            llm = ChatOpenAI(model_name=GPTDict[GPTVersion], 
                            temperature=0, 
                            max_tokens=4000)
            chain = prompt | llm | StrOutputParser()
            for i in range(n_iterations):
                full_response = ""
                st.header("Iteration " + str(i+1) + " from " + st.session_state["article"] + " at " + st.session_state["pdf_url"])
                message_placeholder = st.empty()
                for chunks in chain.stream({"prefix" : prefix, "NQUESTIONS":n_questions, "NCLAIMS":n_claims, "CONTEXT": st.session_state.get("content")}):
                    full_response += (chunks or "")
                    message_placeholder.write(full_response + "▌")
                message_placeholder.write(full_response)
                st.header("", divider = "rainbow")
                st.session_state["Generate"] = False
            
