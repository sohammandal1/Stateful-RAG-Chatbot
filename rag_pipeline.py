# rag_pipeline.py
import torch
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
    """Initializes the Gemma LLM and tokenizer using a token."""
    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME, token=hf_token)
    llm_model = AutoModelForCausalLM.from_pretrained(
        LLM_MODEL_NAME, token=hf_token, device_map="auto", torch_dtype=torch.float16
    )
    text_generation_pipeline = pipeline(
        model=llm_model, tokenizer=tokenizer, task="text-generation",
        return_full_text=False, max_new_tokens=100,
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