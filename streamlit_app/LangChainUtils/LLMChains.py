from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatAnthropic
from langchain_core.output_parsers import StrOutputParser

# Creating a Runnable Branch
# Query -> Decide -> Generate or RAG chain

def DecideAndRunChain(llm):
    decide_prompt = PromptTemplate.from_template("Templates/Decide_Prompts/decide_prompt_00.template")
    decide_chain = decide_prompt | llm | StrOutputParser()
    
    pass

# Creating a QA Runnable Branch
# From question + contextr generate question
def RunQuestionGeneration(llm):
    """The chain takes in the response template which should have the following template.
    {"prefix" : prefix, "NCLAIMS":n_claims, "CONTEXT": article_content}

    Args:
        llm (_type_): LLM in Langchain definition

    Returns:
        chain : QA chain
    """
    response = open("Templates/QA_Generations/response_01.template").read()
    qa_prompt = PromptTemplate.from_template(response)
    qa_chain = qa_prompt | llm | StrOutputParser()
    return qa_chain