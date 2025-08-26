# main.py
import os
import sys
from dotenv import load_dotenv

# Import functions from other modules
from config import VECTOR_STORE_DIR, FILTERED_DOCS_FILEPATH
from data_manager import download_data, filter_and_save_documents
from vector_store_manager import create_vector_store
from rag_pipeline import create_rag_chain
from langchain_core.messages import HumanMessage, AIMessage

def setup_environment():
    """Checks if setup is needed and runs it."""
    if not os.path.exists(VECTOR_STORE_DIR):
        print("--- Initial Setup Required ---")
        if not os.path.exists(FILTERED_DOCS_FILEPATH):
            download_data()
            filter_and_save_documents()
        create_vector_store()
        print("✅ Setup complete.")
    else:
        print("✅ Environment is already set up.")

def main():
    """Main function to run the conversational RAG application."""
    load_dotenv()
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        print("Error: Hugging Face token not found in .env file.", file=sys.stderr)
        sys.exit(1)

    setup_environment()

    print("\nInitializing RAG chain... (This may take a moment)")
    rag_chain = create_rag_chain(hf_token)
    chat_history = []
    print("\n✅ RAG chain initialized. You can start chatting now. Type 'exit' to quit.")

    while True:
        try:
            question = input("\nYou: ")
            if question.lower() == 'exit':
                break

            response = rag_chain.invoke({"input": question, "chat_history": chat_history})
            print(f"\nAI: {response['answer'].strip()}")

            chat_history.extend([
                HumanMessage(content=question),
                AIMessage(content=response["answer"])
            ])
        except KeyboardInterrupt:
            print("\nExiting chat.")
            break
        except Exception as e:
            print(f"An error occurred: {e}", file=sys.stderr)

if __name__ == "__main__":
    main()