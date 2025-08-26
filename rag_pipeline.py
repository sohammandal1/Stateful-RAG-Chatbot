# rag_pipeline.py
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig
from langchain_huggingface import HuggingFacePipeline, HuggingFaceEmbeddings
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_huggingface import HuggingFacePipeline, HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from config import (
    LLM_MODEL_NAME, EMBED_MODEL_NAME, VECTOR_STORE_DIR,
    RETRIEVER_SEARCH_TYPE, RETRIEVER_SEARCH_KWARGS
)

def initialize_llm(hf_token):
    """
    Initializes the Gemma LLM, applying 4-bit quantization only if a
    compatible CUDA GPU is available.
    """
    # --- Updated: Conditional Quantization Logic ---
    if torch.cuda.is_available():
        device = "cuda"
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        print("✅ Using device: CUDA (with 4-bit Quantization)")
    else:
        # Fallback for MPS (Mac) or CPU
        if torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
        quantization_config = None
        print(f"✅ Using device: {device.upper()} (Quantization disabled)")

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME, token=hf_token)

    # Load the model with the dynamic configuration
    llm_model = AutoModelForCausalLM.from_pretrained(
        LLM_MODEL_NAME,
        token=hf_token,
        quantization_config=quantization_config,
        device_map=device if quantization_config else None, # device_map is needed for quantization
        torch_dtype=torch.float16 if device != "cpu" else torch.float32
    )

    # The pipeline will use the device assigned by device_map or run on the specified device
    text_generation_pipeline = pipeline(
        model=llm_model,
        tokenizer=tokenizer,
        task="text-generation",
        # Pass device only if not using device_map (i.e., not quantizing)
        device=None if quantization_config else device,
        return_full_text=False,
        max_new_tokens=100,
    )

    return HuggingFacePipeline(pipeline=text_generation_pipeline)

def create_rag_chain(hf_token):
    """Creates the conversational RAG chain."""
    llm = initialize_llm(hf_token)
    embed_model = HuggingFaceEmbeddings(model_name=EMBED_MODEL_NAME)
    vector_store = Chroma(
        collection_name="simple_wiki_db",
        embedding_function=embed_model,
        persist_directory=VECTOR_STORE_DIR,
    )
    retriever = vector_store.as_retriever(
        search_type=RETRIEVER_SEARCH_TYPE,
        search_kwargs=RETRIEVER_SEARCH_KWARGS
    )

    rephrase_prompt = ChatPromptTemplate.from_messages([
        ("system", "Given a chat history and a follow up question, rephrase the follow up question to be a standalone question."),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])

    history_aware_retriever = create_history_aware_retriever(llm, retriever, rephrase_prompt)

    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an assistant for question-answering tasks. Use the context to answer. If you don't know, say that you don't know. Keep the answer concise.\n\nContext:\n{context}"),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])

    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    return create_retrieval_chain(history_aware_retriever, question_answer_chain)