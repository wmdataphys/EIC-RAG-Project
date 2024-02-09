import streamlit as st
import os
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_community.callbacks import TrubricsCallbackHandler
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnableBranch
from app_utilities import *

SetHeader("AI4EIC-RAG ChatBot")

# Include some explanations

if not st.session_state.get("user_name"):
    st.error("Please login to your account first to further continue.")
    st.stop()

os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
os.environ["TRUBRICS_EMAIL"] = st.secrets["TRUBRICS_EMAIL"]
os.environ["TRUBRICS_PASSWORD"] = st.secrets["TRUBRICS_PASSWORD"]

if st.secrets.get("LANGCHAIN_API_KEY"):
    os.environ["LANGCHAIN_API_KEY"] = st.secrets["LANGCHAIN_API_KEY"]
    os.environ["LANGCHAIN_TRACING_V2"] = st.secrets["LANGCHAIN_TRACING_V2"]
    os.environ["LANGCHAIN_PROJECT"] = st.secrets["LANGCHAIN_PROJECT"]
    os.environ["LANGCHAIN_ENDPOINT"] = st.secrets["LANGCHAIN_ENDPOINT"]
# Creating OpenAIEmbedding()

embeddings = OpenAIEmbeddings()
# Defining some props of DB
SimilarityDict = {"Cosine similarity" : "similarity", "MMR" : "mmr"}

DBProp = {"LANCE" : {"vector_config" : {"db_name" : st.secrets["LANCEDB_DIR"], 
                                        "table_name" : "EIC_archive", 
                                        "embedding_function" : embeddings
                                        },
                     "search_config" : {"metric" : "similarity", 
                                        "search_kwargs" : {"k" : 100}
                                        },
                     "available_metrics" : ["Cosine similarity"]
                     },
          "CHROMA" : {"vector_config" : {"db_name" : st.secrets["CHROMADB_DIR"], 
                                         "embedding_function" : embeddings, 
                                         "collection_name" : "ARXIVS"
                                         },
                      "search_config" : {"metric" : "similarity",
                                         "search_kwargs" : {"k" : 100}
                                         },
                      "available_metrics" : ["Cosine similarity", "MMR"]
                      },
          "PINECONE" : {"vector_config" : {"db_api_key" : st.secrets["PINECONE_API_KEY"], 
                                            "index_name" : "llm-project", 
                                            "embedding_function" : embeddings
                                            },
                        "search_config" : {"metric" : "similarity", 
                                           "search_kwargs" : {"k" : 100}
                                           },
                        "available_metrics" : ["Cosine similarity", "MMR"]
                        },
          }
# Creating retriever

retriever = GetRetriever("PINECONE", DBProp["PINECONE"]["vector_config"], DBProp["PINECONE"]["search_config"])


with st.sidebar:
    if (st.session_state.get("user_name")):
        with st.container():
            st.info("Select VecDB and Properties")
            db_type = st.selectbox("Vector DB", ["PINECONE"])
            similiarty_score = st.selectbox("Retrieval Metric", DBProp[db_type]["available_metrics"])
            max_k = st.select_slider("Max K", options = [10, 20, 30, 40, 50, 100, 150], value = 100)
            if st.button("Select Vector DB"):
                DBProp[db_type]["search_config"]["search_kwargs"]["k"] = max_k
                DBProp[db_type]["search_config"]["metric"] = SimilarityDict[similiarty_score]
                retriever = GetRetriever(db_type, DBProp[db_type]["vector_config"], DBProp[db_type]["search_config"])

if retriever == None:
    st.stop()

from langchain.schema import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough


def format_docs(docs):
    unique_arxiv = list(set(doc.metadata['arxiv_id'] for doc in docs))
    mkdown = """# Retrieved documents \n"""
    for idx, u_ar in enumerate(unique_arxiv):
        mkdown += f"""{idx + 1}. <ARXIV_ID> {u_ar} <ARXIV_ID/> \n
        """
        for i, doc in enumerate(docs):
            if doc.metadata['arxiv_id'] == u_ar:
                mkdown += """  *\t""" + doc.page_content.strip("\n") + " \n"
    return mkdown

llm = ChatOpenAI(model_name="gpt-3.5-turbo-1106", temperature=0, 
                 callbacks=[
                     TrubricsCallbackHandler(
                         project="EIC-RAG-TestRun",
                         config_model={
                             "model" : "gpt-3.5-turbo-1106",
                             "temperature" : 0,
                             "max_tokens" : 4096,
                             "prompt_template" : "rag_prompt_custom",
                         },
                         tags = ["EIC-RAG-TestRun"],
                         user_id = st.session_state["user_name"],
                                             )
                     ], 
                 max_tokens=4096)
decide_response = open("Templates/Decide_Prompts/decide_prompt_00.template", "r").read()
decide_prompt = PromptTemplate.from_template(decide_response)
decide_chain = decide_prompt | llm | StrOutputParser()

response = open("Templates/reponse_01.template", "r").read()
response_rewrite = """\
Follow the instructions very very strictly. Do not add anything else in the response. Do not hallucinate nor make up answers.
- The content below within the tags <MARKDOWN_RESPONSE> and </MARKDOWN_RESPONSE> is presented within a `st.markdown` container in a streamlit chat window. 
- It may have some syntax errors and improperly arranged citations. 
- Strictly do no modify the reference URL nor its text.
- Identify unique reference URL links from the context below and cite them in the form of superscripts.  
- The new citations should be numerical and start from one. There has to be atleast one citation in the response.
- Make sure to only use github flavoured markdown syntax for citations. The superscripts should not be in html tags.
- Check for GitHub flavoured Markdown syntax and Importantly correct the syntax to be compatible with GitHub flavoured Markdown and specifically the superscripts, and arrange the new citations to be numerical starting from one.
- The content may have latex commands as well. Edit them to make it compatible within Github flavoured markdown by adding $ before and after the latex command.
- Make sure the citations are superscripted and has to be displayed properly when presented in a chat window. 
- Do not include the <MARKDOWN_RESPONSE> and <MARKDOWN_RESPONSE/> tags in your answer.
- Strictly do no modify the reference URL nor its text. Strictly have only Footnotes with reference links in style of GithubFlavoured markdown -
- Do not create any additional list of links other than Footnotes in your answer.
<MARKDOWN_RESPONSE>
{markdown_response}
<MARKDOWN_RESPONSE/>
"""
rag_prompt_custom = PromptTemplate.from_template(response)
rag_prompt_rewrite = PromptTemplate.from_template(response_rewrite)


from operator import itemgetter

from langchain.schema.runnable import RunnableMap

rag_chain_from_docs = (
    {
        "context": lambda input: format_docs(input["documents"]),
        "question": itemgetter("question"),
    }
    | rag_prompt_custom
    | llm
    | {"markdown_response" : StrOutputParser()} | rag_prompt_rewrite | llm | StrOutputParser()
)

rag_chain_with_source = RunnableMap(
    {"documents": retriever, "question": RunnablePassthrough()}
) | {
    "answer": rag_chain_from_docs,
}

creative_chain = (
    PromptTemplate.from_template(
        """
        You are an expert in answering questions about Hadronic physics and the upcoming Electron Ion Collider (EIC).
        But remember you do not have upto date information about the project nor you track its updates.
        You will not answer any question that is not related to the EIC or Hadronic physics.
        You will politely decline answering about any other topic. However, to lighten the mood, you will respond with a joke or a quote.
        You are an expert in responding to a question in a professional fashion.
        Starting by greeting and thanking for the question. 
        Answer the question in a very comprehensive way.
        Respond to the question in a fun, calm and professional fashion. 
        Make sure to write a comprehensive answer. 
        End the response with a funny joke or a quote related to the answer. Below is the question you need to respond to.
Question: {question}
"""
    )
    | llm | {"answer" : StrOutputParser()}
)

general_chain = (
    PromptTemplate.from_template(
        """
You are a familiar with the Electron Ion Collider (EIC) and its working. However you are no expert about EIC physics nor have upto date information on it
Respond to the question by starting with saying, you are not sure of the answer but will try to answer at your best.
Answer the question in a very comprehensive way.
<question>
{question}
</question>
"""
    ) | llm | {"answer" : StrOutputParser()}
)

if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-3.5-turbo"

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What is up?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        full_response = ""
        allchunks = None
        with st.spinner("Hmmmm deciding if I need to use Knowledge bank for this query..."):
            outdecide = decide_chain.invoke({"question":prompt})
        if "more info" in outdecide.lower():
            st.info("Gathering info from Knowledge Bank for this query...")
            allchunks = rag_chain_with_source.stream(prompt)
        elif "enough info" in outdecide.lower():
            st.warning("I am going to answer this question with my knowledge.")
            allchunks = creative_chain.stream({"question" : prompt})
        else:
            st.error("I am not sure if I can answer this question. I will try to answer it with my knowledge.")
            allchunks = general_chain.stream({"question": prompt})
        message_placeholder = st.empty()
        for chunk in allchunks:
            full_response += (chunk.get("answer") or "")
            message_placeholder.markdown(full_response + "▌")
        message_placeholder.markdown(full_response)
            #with st.status("Its Feedback time"):
            #    st.
    st.session_state.messages.append({"role": "assistant", "content": full_response})
