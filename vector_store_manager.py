# vector_store_manager.py
import json
from langchain_chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain_huggingface import HuggingFaceEmbeddings
from config import FILTERED_DOCS_FILEPATH, VECTOR_STORE_DIR, EMBED_MODEL_NAME

def create_vector_store():
    """Creates a ChromaDB vector store from the filtered documents."""
    with open(FILTERED_DOCS_FILEPATH, 'r', encoding='utf-8') as f:
        new_documents = json.load(f)

    docs = [Document(page_content=doc) for doc in new_documents]
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=300)
    chunked_docs = splitter.split_documents(docs)
    print(f"Created {len(chunked_docs)} document chunks.")

    embed_model = HuggingFaceEmbeddings(model_name=EMBED_MODEL_NAME)
    print("Creating and persisting the vector store... (This may take a few minutes)")
    Chroma.from_documents(
        documents=chunked_docs,
        embedding=embed_model,
        collection_name="simple_wiki_db",
        persist_directory=VECTOR_STORE_DIR,
    )