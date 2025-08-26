# data_manager.py
import os
import json
import gzip
from sentence_transformers import util
from config import WIKIPEDIA_FILEPATH, FILTERED_DOCS_FILEPATH

def download_data():
    """Downloads the Simple Wikipedia dataset if it doesn't already exist."""
    if not os.path.exists(WIKIPEDIA_FILEPATH):
        print("Downloading Wikipedia dataset...")
        url = f'http://sbert.net/datasets/{os.path.basename(WIKIPEDIA_FILEPATH)}'
        util.http_get(url, WIKIPEDIA_FILEPATH)
        print("Download complete.")

def filter_and_save_documents():
    """Filters documents based on keywords and saves them to a JSON file."""
    testing_words = [
        'india', 'north pole', 'nlp', 'natural language processing', 'linguistics',
        'machine learning', 'artificial intelligence', 'cheetah', 'animal', 'jaguar'
    ]
    new_documents = []
    print("Filtering documents...")
    with gzip.open(WIKIPEDIA_FILEPATH, 'rt', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            doc = data['paragraphs'][0]
            if any(word in doc.lower() for word in testing_words):
                new_documents.append(doc)

    print(f"Found {len(new_documents)} relevant documents.")
    with open(FILTERED_DOCS_FILEPATH, 'w', encoding='utf-8') as f:
        json.dump(new_documents, f)
    print(f"Filtered documents saved to {FILTERED_DOCS_FILEPATH}")