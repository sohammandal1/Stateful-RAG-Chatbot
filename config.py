# config.py
# This file contains configuration constants for the RAG application.

# File and Directory Paths
WIKIPEDIA_FILEPATH = 'simplewiki-2020-11-01.jsonl.gz'
FILTERED_DOCS_FILEPATH = 'filtered_documents.json'
VECTOR_STORE_DIR = "./simple_wiki_db"

# Model Names
EMBED_MODEL_NAME = 'thenlper/gte-base'
LLM_MODEL_NAME = "google/gemma-2-2b-it"

# Retriever Settings
RETRIEVER_SEARCH_TYPE = "similarity_score_threshold"
RETRIEVER_SEARCH_KWARGS = {"k": 5, "score_threshold": 0.2}