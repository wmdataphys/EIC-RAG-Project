from openai import OpenAI
import streamlit as st


import lancedb
from langchain.vectorstores import LanceDB
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
import os

os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
db = lancedb.connect("../my-app/lancedb_meta_data")
table = db.open_table("EIC_archive")
embeddings = OpenAIEmbeddings()
vectorstore = LanceDB(connection = table, embedding = embeddings)

retriever = vectorstore.as_retriever(search_type = "similarity", search_kwargs={"k" : 100})



llm = ChatOpenAI(model_name="gpt-3.5-turbo-1106", temperature=0)

from langchain.schema import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough


def format_docs(docs):
    return f"\n\n".join(f'{i+1}. ' + doc.page_content for i, doc in enumerate(docs))


from langchain.prompts import PromptTemplate

template2 = """\
You are an expert in providing up to date information about the Electron Ion Collider (EIC), tasked with answering any question \
about EIC based only on the provided context.

Generate a comprehensive, and informative answer of 100 words or less for the \
given question based solely on the provided search results (URL and content). You must \
only use information from the provided search results. Use an unbiased and \
journalistic tone. Combine search results together into a coherent answer. Do not \
repeat text. You should use bullet points in your answer for readability. Make sure to break down your answer into bullet points.\
You should not hallicunate nor build up any references, Use only the `context` html block below and its associated `ARXIV_ID` if you find the context relevant. 
Make sure not to repeat the same context. Be specific to the exact question asked for.\
After each bullet point, cite up to 5 most relavant arxiv_id associated with the `context` html block from which the bullet point was generated. \
The citations should be taken from the context between the tags <ARXIV_ID> and <ARXIV_ID/>. Only quote relavant arxiv_id \

Only quotee the most relevant arxiv_id that you find from `context` block. Note that the contexts are numbered according to the cosine similarity index.\
Place these citations at the end of the sentence or paragraph that reference them. If different results refer to different entities within the same name, write separate \
answers for each entity.

If there is nothing in the context relevant to the question at hand, just say "Hmm, \
I'm not sure." Don't try to make up an answer. Write the answer in the form of markdown bullet points.\
Make sure to highlight the most important key words in red color. Dot repeat any context nor points in the answer.\

Anything between the following `context`  html blocks is retrieved from a knowledge \
bank, not part of the conversation with the user. The context are numbered based on its knowledge retrival and increasing cosine similarity index. \
After each context the arxiv_id is given within the <ARXIV_ID> html block. \
Make sure to consider the order in which they appear context appear. It is an increasing order of cosine similarity index.\
The contents are formatted in latex, you need to remove any special characters and latex formatting before cohercing the points to build your answer.\
You will cite no more than 5 citations from the context below.\
Make sure these citations have to relavant as well as not repetitive in nature.

<context>
    {context} <ARXIV_ID> {arxiv_id} <ARXIV_ID/>
<context/>

REMEMBER: If there is no relevant information within the context, just say "Hmm, I'm \
not sure." Don't try to make up an answer. Anything between the preceding 'context' \
html blocks is retrieved from a knowledge bank, not part of the conversation with the \
user.\
Question: {question}
"""
rag_prompt_custom = PromptTemplate.from_template(template2)

from operator import itemgetter

from langchain.schema.runnable import RunnableMap

rag_chain_from_docs = (
    {
        "context": lambda input: format_docs(input["documents"]),
        "arxiv_id": lambda input: [doc.metadata['arxiv_id'] for doc in input["documents"]],
        "question": itemgetter("question"),
    }
    | rag_prompt_custom
    | llm
    | StrOutputParser()
)
rag_chain_with_source = RunnableMap(
    {"documents": retriever, "question": RunnablePassthrough()}
) | {
    "documents": lambda input: [doc.metadata['arxiv_id'] for doc in input["documents"]],
    "answer": rag_chain_from_docs,
}

st.title("ChatGPT-like clone")

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

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
        message_placeholder = st.empty()
        full_response = ""
        for chunk in rag_chain_with_source.stream(prompt):
            full_response += (chunk.get("answer") or "")
            message_placeholder.markdown(full_response + "▌")
        message_placeholder.markdown(full_response)
    st.session_state.messages.append({"role": "assistant", "content": full_response})
