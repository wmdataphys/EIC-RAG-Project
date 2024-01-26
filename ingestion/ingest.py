import os, glob, arxiv, argparse
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter

def SaveFromPDF(args):
    DB_NAME = args.db_name
    SRC_DIR = args.src_dir
    COLLECTION_NAME = args.table_name
    OPENAI_API_KEY = args.openai_api_key
    CHUNKING_TYPE = args.chunking
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
    embeddings = OpenAIEmbeddings()
    AllPDFs = [os.path.join(f, f.split("/")[-1] + ".pdf") for f in glob.glob(SRC_DIR + "/*")]
    for pdf in AllPDFs:
        print ("Processing: " + pdf)
        
        loader = PyPDFLoader(pdf)
        data = loader.load_and_split()
        arxiv_id = pdf.split("/")[-1].split(".pdf")[0]
        
        search = arxiv.Search(id_list=[arxiv_id])
        paper = next(arxiv.Client().results(search))
        meta_data = {"arxiv_id": paper.entry_id, 
            "title": paper.title, 
            "categories" : '\n'.join([f'{i+1}. {cat}' for i, cat in enumerate(paper.categories)]),
            "primary_category": paper.primary_category,
            "published": str(paper.published),
            "authors": '\n'.join([f'{i+1}. {auth.name}' for i, auth in enumerate(paper.authors)])
            }
        print ("Title is :" + """{}""".format(meta_data["title"]))
        for page in data:
            print ("Processing page " + str(page.metadata['page']))
            texts = text_splitter.create_documents([page.page_content], metadatas = [meta_data])
            Chroma.from_documents(texts, embeddings, persist_directory=DB_NAME, collection_name=COLLECTION_NAME)

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="Create Vector DB and perform ingestion from source files")
    argparser.add_argument('-s', '--src_dir', type=str, required=True, help = "Source directory where arxiv sources are stored")
    argparser.add_argument('-db', '--db_name', type=str, required=True, help = "Name of the LanceDB database to be created")
    argparser.add_argument('-t', '--table_name', type=str, required=False, help = "Name of the LanceDB table to be created", default = "EIC_archive")
    argparser.add_argument('-openai_key', '--openai_api_key', type=str, required=True, help = "OpenAI API key")
    argparser.add_argument('-c', '--chunking', type = str, required=False, help = "Type of Chunking PDF or LATEX", default = "PDF")
    argparser.add_argument('-n', '--nthreads', type=int, default=-1)
    
    args = argparser.parse_args()
    SaveFromPDF(args)
    
    
    
    