from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader

import streamlit as st
import numpy as np
import arxiv, os

from app_utilities import num_tokens_from_prompt, SetHeader
from langchain import callbacks
from langsmith import Client

from LangChainUtils.LLMChains import RunQuestionGeneration

os.environ["LANGCHAIN_PROJECT"] = st.secrets["LANGCHAIN_EVAL_PROJECT"]
os.environ["LANGCHAIN_RUN_NAME"] = "QA Generation"

GPT_CONTEXT_LEN = 27000 # 32k - 4096
CHAR_PER_TOKEN = 4
WORD_LIM = GPT_CONTEXT_LEN * CHAR_PER_TOKEN

client = Client()
SetHeader("RAG Generate Questions")
# Some explanations to do 

st.markdown(open("streamlit_app/Resources/Markdowns/QA_Generation.md", "r").read())

if not st.session_state.get("user_name"):
    st.error("Please login to your account first to further continue and generate questions.")
    st.stop()

# mode = 0 for user, 1 for annotator, 2 for developer   
if st.session_state.get("user_mode", -1) > 0:
    with st.sidebar:
        st.toggle("Contribute to Evaluation", value = False, key = "contribute_to_eval", help="Toggle this to contribute to evaluation for each response")
    if st.session_state.contribute_to_eval:
        with st.container(border = True):
            st.title("Langsmith details")
            st.markdown("## PROJECT NAME: " + os.environ["LANGCHAIN_PROJECT"])
            st.markdown("## USER: " + st.session_state.get("user_name"))
            st.markdown("## RUN NAME: " + os.environ["LANGCHAIN_RUN_NAME"])

# Start by defining article load as false
article_keys = ["article_loaded", "article_primary_category", 
                "article_categories", "article_title", 
                "article_abstract", "article_url", "article_doi", 
                "article_authors", "article_content", "article_id",
                "article_date", "article_num_tokens", "article_pages",
                "full_content"
                ]

if "load_article" not in st.session_state:
    st.session_state.load_article = False


articles = open("streamlit_app/Resources/ARXIV_SOURCES.info", "r").readlines()  

def LoadArticle(Load: bool, article_keys: list):
    for key in article_keys:
        st.session_state[key] = None
    st.session_state.questions = []
    st.session_state.load_article = Load
    st.session_state.generation_count = 0

st.header("", divider = "rainbow")
col_Al, col_A, colAr = st.columns([1, 4, 1])
with col_A:
    st.header("Load an Article from arxiv database to generate questions")
    st.button("Load an article from arxiv..", on_click = LoadArticle, args = [True, article_keys])
st.header("", divider = "rainbow")
if st.session_state.load_article:
    with st.status("Loading article...", expanded = True, state = "running") as status:
        st.write("Selecting a random article....")
        article = np.random.choice(articles).strip("\n")
        st.write(f"Searching {article} ID from arxiv.org...")
        try:
            search = arxiv.Search(id_list=[article])
            paper = next(arxiv.Client().results(search))
        except:
            status.update(label = "Unable to load article. Please try again.", state = "error", expanded = True)
            st.stop()
        st.session_state["article_id"] = article
        st.session_state["article_title"] = paper.title
        st.session_state["article_abstract"] = paper.summary
        st.session_state["article_url"] = paper.pdf_url
        st.session_state["article_doi"] = paper.doi
        st.session_state["article_authors"] = '\n'.join([f'{i+1}. {auth.name}' for i, auth in enumerate(paper.authors)])
        st.session_state["article_date"] = paper.published
        st.session_state["article_primary_category"] = paper.primary_category
        st.session_state["article_categories"] = ', '.join([f'{cat}' for i, cat in enumerate(paper.categories)])
        st.write(f"Loading article {article} in memory...")
        docs = PyPDFLoader(paper.pdf_url).load()
        st.session_state.article_pages = len(set([doc.metadata.get('page') for doc in docs]))
        full_content = "\n".join([doc.page_content for doc in docs])
        if (len(full_content) > WORD_LIM):
            status.update(label = f"Article {article} is too large to load in memory. Chunking it smaller to fit it.", state = "running", expanded = True)
            st.warning("Too large article to load in memory. Chunking it smaller to fit it.")
            start = np.random.randint(0, len(full_content) - WORD_LIM)
            content = full_content[start:start + WORD_LIM]
        else:
            content = full_content
        st.session_state["full_content"] = full_content
        st.session_state["article_content"] = content
        st.write(f"Article {article} successfully loaded in memory. Precounting tokens now...")
        num_tokens = num_tokens_from_prompt(content, "gpt-3.5-turbo-1106")
        st.session_state["article_num_tokens"] = num_tokens
        st.session_state.article_loaded = True
        st.session_state.load_article = False
        status.update(label = f"Article {article} Successfully loaded and ready, Tokens = {num_tokens}!", state = "complete", expanded = False)

if not st.session_state.get("article_id"):
    st.stop()

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Article ID", divider = "rainbow")
    st.write(st.session_state.get("article_id", ""))
    st.subheader("Paper Summary", divider = "rainbow")
    st.write(st.session_state.get("article_abstract", ""))
    st.subheader("Publication Date", divider = "rainbow")
    st.write(st.session_state.get("article_date", ""))
    st.subheader("Primary Category", divider = "rainbow")
    st.write(st.session_state.get("article_primary_category", ""))
    
with col2:
    st.subheader("Paper Title", divider = "rainbow")
    st.write(st.session_state.get("article_title", ""))
    st.subheader("Paper Authors", divider = "rainbow")
    st.write(st.session_state.get("article_authors", ""))
    st.subheader("Link to PDF", divider = "rainbow")
    st.write(st.session_state.get("article_url", ""))
    st.subheader("Published journal (if any)", divider = "rainbow")
    st.write(st.session_state.get("article_journal", "Not Published Yet"))
    st.subheader("Categories", divider = "rainbow")
    st.write(st.session_state.get("article_categories", ""))
    st.subheader("Num of Pages and tokens", divider = "rainbow")
    st.markdown(r"__Pages__: " + str(st.session_state.get("article_pages", "")) + r"  &  __Tokens__: " + str(st.session_state.get("article_num_tokens", "")))

if st.session_state.get("article_loaded") and len(st.session_state.get("article_content", [])) < WORD_LIM:
    article_container = st.container(border = True)
    article_container.subheader("Article Content", divider = "rainbow")
    with st.expander("Expand to show", expanded = False):
        st.write(st.session_state.get("article_content", ""))

GPTDict = {"3.5": "gpt-3.5-turbo-1106", "4": "gpt-4-0125-preview"}
prefix = open("Templates/QA_Generations/example_01.template").read()

def gen_submit(generate: bool):
    st.session_state.response_container = st.empty()
    st.session_state["Generate"] = generate
    st.session_state.generation_count += 1

for ques in st.session_state.get("questions", []):
    with st.expander("Question - " + ques["qnum"], expanded = False):
        st.header("", divider = "rainbow")
        st.subheader(ques["question"])
        st.subheader("Answer")
        st.write(ques["answer"])
        if (len(ques["content"]) > WORD_LIM):
            st.subheader("Content")
            st.write(ques["content"])
        st.markdown(r"""<h2 style="text-align: center;">Link to trace [🛠️]""" + ques["trace_link"] + "</h2>")
        st.header("", divider = "rainbow")
        

with st.container(border = True):
    st.title("Lets Start Generating Questions")
    with st.form("Generate Question"):
        f_col1, f_col2, f_col3 = st.columns([1, 1, 1])
        with f_col1:
            n_questions = st.number_input("Number of questions to be generated", min_value = 1, max_value = 5)
        with f_col2:
            n_claims = st.number_input("Number of claims to be generated in each question", min_value = 1, max_value = 10)
            st.form_submit_button("Generate", on_click = gen_submit, args = (True,))
        with f_col3:
            GPTVersion = st.selectbox("GPT Version", ["4"])
             
    if st.session_state.get("Generate"):
        st.session_state["Generate"] = False
        llm = ChatOpenAI(model_name=GPTDict[GPTVersion], 
                         temperature=0, 
                         max_tokens=4000
                         )
        chain = RunQuestionGeneration(llm).with_config({"run_name" : os.environ["LANGCHAIN_RUN_NAME"]
                                }
                               )
        if (len(st.session_state["full_content"]) > WORD_LIM):
            _start = np.random.randint(0, len(st.session_state["full_content"]) - WORD_LIM)
            st.session_state["article_content"] = st.session_state["full_content"][_start:_start + WORD_LIM]
        for i in range(n_questions):
            full_response = ""
            st.header("Question " + str(i+1) + " from " + st.session_state["article_id"] + " at " + st.session_state["article_url"])
            message_placeholder = st.empty()
            metadata = {"username": st.session_state.user_name, 
                        "article_id": st.session_state["article_id"],
                        "article_url": st.session_state["article_url"],
                        "claims" : n_claims
                        }
            tags = [f"claims-{n_claims}", st.session_state["article_id"], GPTVersion]
            with callbacks.collect_runs() as cb:
                for chunks in chain.stream({"prefix" : prefix, "NCLAIMS":n_claims, 
                                            "CONTEXT": st.session_state.get("article_content")
                                            }, 
                                           {"metadata": metadata, 
                                            "tags": tags
                                            }
                                           ):
                    full_response += (chunks or "")
                    message_placeholder.write(full_response + "▌")
                st.session_state.DataGen_run_id = cb.traced_runs[0].id
                st.session_state.run_url = client.read_run(st.session_state.DataGen_run_id).url
            message_placeholder.write(full_response) 
            st.markdown(r"""<h2 style="text-align: center;">Link to trace [🛠️]""" + f"({st.session_state.run_url})" + "</h2>", unsafe_allow_html=True)
            st.header("", divider = "rainbow")
            st.session_state.questions.append({"qnum" : f"Gen: {st.session_state.generation_count}, Q: {i}", 
                                               "content" : st.session_state["article_content"],
                                               "question": full_response.split("A:")[0], 
                                               "answer": full_response.split("A:")[-1],
                                               "trace_link": st.session_state.run_url
                                               }
                                              )
            
