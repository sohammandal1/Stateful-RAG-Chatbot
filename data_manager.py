# data_manager.py
import os
import json
import gzip
import requests
from tqdm import tqdm
from config import WIKIPEDIA_FILEPATH, FILTERED_DOCS_FILEPATH

# --- Updated download_data function ---
def download_data():
    """
    Downloads the Simple Wikipedia dataset using requests and shows a
    progress bar with tqdm.
    """
    if not os.path.exists(WIKIPEDIA_FILEPATH):
        print("Downloading Wikipedia dataset...")
        url = f'http://sbert.net/datasets/{os.path.basename(WIKIPEDIA_FILEPATH)}'

        # Use requests to stream the download
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Raise an exception for bad status codes

        # Get the total file size from the headers
        total_size_in_bytes = int(response.headers.get('content-length', 0))

        # Set up the tqdm progress bar
        progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)

        # Write the file out in chunks
        with open(WIKIPEDIA_FILEPATH, 'wb') as f:
            for chunk in response.iter_content(chunk_size=1024):
                progress_bar.update(len(chunk))
                f.write(chunk)
        
        progress_bar.close()

        if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
            print("ERROR, something went wrong during download.")

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
