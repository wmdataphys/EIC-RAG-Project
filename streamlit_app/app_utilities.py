import os, sqlite3, lancedb, tiktoken, bcrypt
from pinecone import Pinecone, ServerlessSpec
from enum import Enum
from langchain_community.vectorstores import LanceDB, Chroma
from langchain_community.vectorstores import Pinecone as LangPinecone

class UserNotFoundError(Exception):
    pass

class DBNotFoundError(Exception):
    pass
def hash_password(password: str):
    bytes = password.encode('utf-8')
    salt = bcrypt.gensalt()
    return bcrypt.hashpw(bytes, salt)
def get_user_info(db_name, username):
    if not os.path.exists(db_name):
        raise FileNotFoundError(f"Database {db_name} does not exist.")
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    cursor.execute('''
        SELECT username, first_name, last_name, password FROM users WHERE username = ?
    ''', (username,))
    user = cursor.fetchone()
    conn.close()
    if user:
        return user
    else:
        return None

class VectorDB(Enum):
    LANCE = 1
    CHROMA = 2
    PINECONE = 3
    

def GetRetriever(TYPE: str, vector_config: dict, search_config = {}):
    if TYPE == VectorDB.LANCE.name:
        db = lancedb.connect(vector_config["db_name"])
        table = db.open_table(vector_config["table_name"])
        return LanceDB(connection = table, 
                       embedding = vector_config["embedding_function"]
                       ).as_retriever(search_type = search_config.get("metric", "similarity"), 
                                      search_kwargs=search_config.get("search_kwargs", {"k" : 100}) 
                                      )
    elif TYPE == VectorDB.CHROMA.name:
        return Chroma(persist_directory = vector_config["db_name"], 
                      embedding_function = vector_config["embedding_function"], 
                      collection_name=vector_config["collection_name"]
                      ).as_retriever(search_type = search_config.get("metric", "similarity"), 
                                     search_kwargs=search_config.get("search_kwargs", {"k" : 100})
                                     )
    elif TYPE == VectorDB.PINECONE.name:
        pc = Pinecone(api_key = vector_config["db_api_key"])
        if vector_config["index_name"] not in pc.list_indexes().names():
            raise DBNotFoundError(f"Database {vector_config['index_name']} does not exist.")
        return LangPinecone.from_existing_index(vector_config["index_name"], 
                                                    vector_config["embedding_function"]
                                                    ).as_retriever(search_type = search_config.get("metric", "similarity"), 
                                                                   search_kwargs=search_config.get("search_kwargs", {"k" : 100})
                                                                   )
    else:
        raise NotImplementedError("Invalid VectorDB type")

def num_tokens_from_prompt(prompt: str, model: str) -> int:
        """Return the number of tokens used by a prompt."""
        encoding = tiktoken.encoding_for_model(model)
        return len(encoding.encode(prompt))
def num_tokens_from_messages(messages, model) -> int:
        """Return the number of tokens used by a list of messages."""
        encoding = tiktoken.encoding_for_model(model)
        tokens_per_message = 4  # every message follows <|start|>{role/name}\n{content}<|end|>\n
        tokens_per_name = 1
        num_tokens = 0
        for message in messages:
            num_tokens += tokens_per_message
            for key, value in message.items():
                num_tokens += len(encoding.encode(value))
                if key == "name":
                    num_tokens += tokens_per_name
        num_tokens += 4  # every reply is primed with <|start|>assistant<|message|>
        return num_tokens