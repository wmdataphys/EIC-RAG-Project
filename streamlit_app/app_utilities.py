import os, sqlite3, lancedb
from enum import Enum
from langchain_community.vectorstores import LanceDB, Chroma

class UserNotFoundError(Exception):
    pass

class DBNotFoundError(Exception):
    pass

def get_user_info(db_name, username):
    if not os.path.exists(db_name):
        raise FileNotFoundError(f"Database {db_name} does not exist.")
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    cursor.execute('''
        SELECT username, first_name, last_name FROM users WHERE username = ?
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
    else:
        raise NotImplementedError("Invalid VectorDB type")