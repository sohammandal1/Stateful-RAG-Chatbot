# Conversational RAG Chatbot with TinyLlama & Gemma

This project is a sophisticated, yet easy-to-run Conversational RAG (Retrieval-Augmented Generation) chatbot. It uses a local vector database built from a subset of Wikipedia articles to answer questions on topics like Machine Learning, AI, India, and various animals. The application features an interactive web UI and is designed to be highly modular and portable across different hardware.



## ✨ Features

* **Conversational Memory:** The chatbot remembers the context of the conversation to answer follow-up questions accurately.
* **Retrieval-Augmented Generation (RAG):** Reduces model hallucinations by fetching relevant information from a trusted knowledge base (ChromaDB) before generating an answer.
* **Interactive Web UI:** A user-friendly and responsive chat interface built with Gradio.
* **Automated Hardware Detection:** Intelligently detects and utilizes the best available hardware (**NVIDIA GPU (CUDA)**, **Apple Silicon (MPS)**, or **CPU**).
* **Conditional Quantization:** Automatically applies 4-bit quantization when an NVIDIA GPU is detected, significantly boosting performance and reducing memory usage.
* **Modular Codebase:** The project is organized into logical modules for configuration, data management, vector store creation, and the RAG pipeline, making it easy to understand and extend.

---

## 🛠️ How It Works (Architecture)

The application follows a complete RAG pipeline from data ingestion to user interaction:

1.  **One-Time Setup (Automated):**
    * **Data Ingestion:** On the first run, the script downloads a subset of the Simple Wikipedia dataset.
    * **Filtering:** It filters these articles for documents relevant to a predefined set of keywords (e.g., 'machine learning', 'cheetah').
    * **Vector Store Creation:** The filtered documents are chunked and embedded using the `thenlper/gte-base` model. These embeddings are then stored locally in a persistent ChromaDB vector store.

2.  **Runtime Chat Logic:**
    * When you ask a question, the `history_aware_retriever` first determines if it's a follow-up question and rephrases it to be standalone if needed.
    * The rephrased question is used to retrieve the most relevant document chunks from the ChromaDB vector store.
    * These chunks are injected as context into a prompt, along with your original question and the chat history.
    * The complete prompt is sent to the Language Model (default: `TinyLlama-1.1B`) to generate a factually grounded answer.
    * The response is streamed back to the Gradio UI for a real-time chat experience.

---

## 🚀 Getting Started

Follow these steps to get the chatbot running on your local machine.

### Prerequisites

* Python 3.10 or higher
* Git

### 1. Clone the Repository

Open your terminal and clone the repository to your local machine:
```bash
git clone https://github.com/sohammandal1/Stateful-RAG-Chatbot
cd Stateful-RAG-Chatbot
```

### 2. Set Up the Python Environment

It's highly recommended to use a virtual environment.
```bash
# Create a virtual environment
python -m venv venv

# Activate it
# On macOS/Linux:
source venv/bin/activate
# On Windows:
.\venv\Scripts\activate
```

### 3. Install Dependencies

Install all the required libraries from the `requirements.txt` file.
```bash
pip install -r requirements.txt
```

### 4. Configure Your Hugging Face Token

The application needs a Hugging Face token to download the language model.

1.  Create a file named `.env` in the root of the project directory.
2.  Add your Hugging Face access token to this file. You can get a token from [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).
    ```
    # .env
    HF_TOKEN="your_huggingface_access_token_here"
    ```

### 5. Run the Application

Now you're ready to launch the chatbot!
```bash
python app.py
```
* **On the first run**, the script will perform the one-time setup (downloading data, creating the vector store). This might take several minutes depending on your internet speed and computer.
* **On subsequent runs**, the script will detect that the setup is complete and launch the UI almost instantly.

Once it's running, open the local URL shown in your terminal (e.g., `http://127.0.0.1:7860`) in your web browser.

---

## 💡 Performance & Customization

* **Hardware:** Performance is highly dependent on your hardware. An NVIDIA GPU (CUDA) will provide the best experience. The app will run on an Apple Silicon Mac (MPS) or a CPU, but with higher latency.
* **Changing the Model:** You can easily switch the language model by editing the `LLM_MODEL_NAME` variable in the **`config.py`** file. For example, to use the more powerful (but much more demanding) Gemma model, change it to:
    ```python
    # in config.py
    LLM_MODEL_NAME = "google/gemma-2-2b-it"
    ```
    **Note:** The Gemma 2B model requires a powerful GPU and may not run smoothly on most local machines.

## 💻 Technologies Used

* **Core Logic:** LangChain, PyTorch
* **LLM & Embeddings:** Hugging Face Transformers
* **Vector Database:** ChromaDB
* **UI:** Gradio
* **Models:** `TinyLlama/TinyLlama-1.1B-Chat-v1.0` (default), `thenlper/gte-base` (embeddings)