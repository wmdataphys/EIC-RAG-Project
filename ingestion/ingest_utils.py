import os, time


def PINECONE_config():
    return {
        "vector_config" : {"db_name" : os.environ["PINECONE_DIR"],
                           "table_name" : "EIC_archive",
                           "embedding_function" : embeddings
                           },
        "search_config" : {"metric" : "similarity",
                           "search_kwargs" : {"k" : 100}
                           }
    }