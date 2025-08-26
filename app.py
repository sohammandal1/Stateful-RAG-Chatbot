# app.py
import os
import sys
import gradio as gr
from dotenv import load_dotenv

# Import functions from other modules
from config import VECTOR_STORE_DIR, FILTERED_DOCS_FILEPATH
from data_manager import download_data, filter_and_save_documents
from vector_store_manager import create_vector_store
from rag_pipeline import create_rag_chain
from langchain_core.messages import HumanMessage, AIMessage

# --- 1. SETUP AND INITIALIZATION ---

# Load environment variables and perform initial setup
print("Performing initial checks and setup...")
load_dotenv()
hf_token = os.getenv("HF_TOKEN")
if not hf_token:
    print("Error: Hugging Face token not found in .env file.", file=sys.stderr)
    sys.exit(1)

if not os.path.exists(VECTOR_STORE_DIR):
    print("--- Initial Setup Required ---")
    if not os.path.exists(FILTERED_DOCS_FILEPATH):
        download_data()
        filter_and_save_documents()
    create_vector_store()
    print("✅ Setup complete.")
else:
    print("✅ Environment is already set up.")

# Initialize the RAG chain (this can take a moment)
print("\nInitializing RAG chain... (This may take a moment)")
rag_chain = create_rag_chain(hf_token)
print("✅ RAG chain initialized.")

# --- 2. GRADIO LOGIC ---

def format_chat_history(chat_history):
    """Formats Gradio's history to match LangChain's expected format."""
    formatted_history = []
    for user_msg, ai_msg in chat_history:
        formatted_history.append(HumanMessage(content=user_msg))
        formatted_history.append(AIMessage(content=ai_msg))
    return formatted_history

def chat_response(message, history):
    """
    This is the core function that Gradio calls.
    It accumulates tokens and yields the full response at each step
    to create a typewriter effect.
    """
    langchain_history = format_chat_history(history)
    response_generator = rag_chain.stream({"input": message, "chat_history": langchain_history})

    full_response = ""
    for chunk in response_generator:
        if "answer" in chunk:
            full_response += chunk["answer"]
            yield full_response


# --- 3. LAUNCH THE UI ---
ui = gr.ChatInterface(
    fn=chat_response,
    title="Chat with Your Documents 📄",
    description="This is a conversational AI powered by Retrieval-Augmented Generation (RAG). Ask a question, and the chatbot will answer based on the knowledge from its document store.",
    examples=[
        ["What is Retrieval-Augmented Generation?"],
        ["Can you summarize the key points about [your topic]?"],
        ["Tell me more about your previous answer."]
    ]
)

# Launch the web server
print("Launching Gradio UI...")
ui.launch(share=True)