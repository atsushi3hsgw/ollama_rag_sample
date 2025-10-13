import os
from dotenv import load_dotenv
import logging

from typing import List, Dict, Any, Optional
import yaml

import streamlit as st

from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.embeddings import Embeddings

from langchain_core.vectorstores import VectorStore
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.faiss import DistanceStrategy

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableMap
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import AIMessage, HumanMessage

from ollama_client import get_available_models, OLLAMA_URL

RAG_INDEX_DIR = os.getenv("RAG_INDEX_DIR", ".rag_index_dir")
RAG_INDEX_PATH = os.path.join(RAG_INDEX_DIR, "rag_index")
PROMPTS_PATH = os.getenv("PROMPTS_PATH", "prompts_langchain.yml")

load_dotenv()
def setup_logger(name=__name__):
    """
    Setup logger with environment variable LOG_LEVEL
    """
    logger = logging.getLogger(name)
    if not logger.hasHandlers():
        level_str = os.getenv("LOG_LEVEL", "INFO").upper()
        level = getattr(logging, level_str, logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '[%(asctime)s] %(levelname)s in %(name)s: %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(level)
        logger.info(f"Logger initialized with level: {level_str}")
    return logger

logger = setup_logger("app_langchain")

@st.cache_data
def get_model_list() -> List[str]:
    """
    Get available models from Ollama
    """
    return get_available_models()

def load_embedding_model() -> Embeddings:
    """
    Cache and load embedded models
    """
    _model_name = os.getenv("EMBEDDING_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    logger.info(f"Loading embedding model: {_model_name}")
    embedding = HuggingFaceEmbeddings(model_name=_model_name
    , encode_kwargs={'normalize_embeddings': True})
    # The default cache directory is ~/.cache/huggingface/hub
    # You can change it by setting the environment variable HUGGINGFACE_HUB_CACHE
    logger.info("Embedding model loaded")
    return embedding

def init_vector_store(_embedder) -> VectorStore:
    """
    Load FAISS vector store if exists
    """
    if os.path.exists(RAG_INDEX_PATH):
        logger.info(f"Loading vector store from {RAG_INDEX_PATH}")
        vector_store = FAISS.load_local(RAG_INDEX_PATH, _embedder
        , allow_dangerous_deserialization=True
        , distance_strategy = DistanceStrategy.MAX_INNER_PRODUCT
        , normalize_L2 = False)
    else:
        logger.info("No existing vector store found, initializing a new one")
        # FAISS requires at least one vector to initialize
        vector_store = FAISS.from_texts(["hello world"], _embedder
        , distance_strategy = DistanceStrategy.MAX_INNER_PRODUCT
        , normalize_L2 = False)
    
    logger.info("Vector store initialized")
    return vector_store
            
def retrieve_with_score(vector_store: VectorStore, query: str, k: int=5, **kwargs: any)-> List[Document]:
    """
    Retrieve documents with similarity scores.
    When creating a Retriever with as_retriever(), the document score cannot be retrieved.
    """
    
    logger.debug(f"Retrieving top {k} documents for query: {query}")
    logger.debug(f"Additional kwargs: {kwargs}")
    
    # Some VectorStores have similarity_search_with_score(), others do not
    if hasattr(vector_store, "similarity_search_with_score"):
        results = vector_store.similarity_search_with_score(
            query, k=k, **kwargs
        )
    else:
        # Fallback: get docs only, then simulate score = 1.0
        docs = vector_store.similarity_search(query, k=k, **kwargs)
        results = [(doc, 1.0) for doc in docs]
    
    docs = []
    for doc, score in results:
        doc.metadata["score"] = score
        docs.append(doc)
    
    logger.debug(f"Retrieved {len(docs)} documents")
    logger.debug(f"Documents: {docs}")
    
    return docs

def format_docs(docs: List[Document]) -> str:
    """
    Format documents for prompt context
    """
    if not docs:
        return "No relevant context found."
    formatted = []
    for doc in docs:
        source = os.path.basename(doc.metadata.get("source", "NA"))
        score = doc.metadata.get("score", None)
        score_str = f" (score={score:.3f})" if score is not None else ""
        formatted.append(f"Source: {source}{score_str}\nContent: {doc.page_content}")
    return "\n\n".join(formatted)

def build_rag_chain(messages:List[tuple[str, str]], vector_store: VectorStore, threshold: int, llm: BaseChatModel):
    """
    Build a complete RAG (Retrieval-Augmented Generation) chain.
    Combines retriever, formatter, prompt, and LLM.
    """
    prompt = ChatPromptTemplate.from_messages(messages)
    chain = RunnableMap({
        "context": (lambda x: retrieve_with_score(vector_store, x["query"], score_threshold=threshold)), 
        "query": RunnablePassthrough(),
    }) | RunnableMap({
        "context": (lambda x: format_docs(x["context"])),
        "query": (lambda x: x["query"]["query"]),
    })| prompt | llm | StrOutputParser()
    return chain

def build_query_expansion_chain(messages:List[tuple[str, str]], llm: BaseChatModel):
    """
    Build a query-expansion chain.
    Expands user queries semantically while preserving meaning.
    """
    prompt = ChatPromptTemplate.from_messages(messages)
    chain = prompt | llm | StrOutputParser()
    return chain

def load_prompt_messages(file_path: str) -> dict[str, list[tuple[str, str]]]:
    """
    Loads a prompt message from a YAML file, converts it to LangChain format [(role, content), ...], and returns it.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        data: dict[str, Any] = yaml.safe_load(f)

    prompts: dict[str, list[tuple[str, str]]] = {}
    for key, messages in data.items():
        prompts[key] = [(m["role"], m["content"]) for m in messages]

    logger.info(f"Loaded prompts from {file_path}: {list(prompts.keys())}")
    logger.debug(f"Prompts detail: {prompts}")
    
    return prompts

if "llm" not in st.session_state:
    st.session_state.llm = None
    st.session_state.model = None

if "vector_store" not in st.session_state:
    embedding = load_embedding_model()
    st.session_state.vector_store = init_vector_store(embedding)

if "prompts" not in st.session_state:
    st.session_state.prompts = load_prompt_messages(PROMPTS_PATH)

query_expansion_messages = st.session_state.prompts.get("query_expansion_messages", [
    ("system", "You are a helpful assistant that expands user queries semantically while preserving the original meaning."),
    ("user", "Expand the following query: {query}")
])
rag_prompt_messages = st.session_state.prompts.get("rag_prompt_messages", [
    ("system", "You are a helpful assistant that answers questions based on the provided context."),
    ("user", "Use the following context to answer the question. If the context is not relevant, just say you don't know.\n\nContext:\n{context}\n\nQuestion: {query}")
])
simple_prompt_messages = st.session_state.prompts.get("simple_prompt_messages", [
    ("system", "You are a helpful assistant that answers questions based on your knowledge."),
    ("user", "Answer the following question: {query}")
])

st.title("RAG Chatbot with File Upload (LangChain ver.)")

with st.sidebar:
    st.header("Configs")

    model_list = get_model_list()
    selected_model = st.selectbox("model", model_list)

    use_context = st.checkbox("RAG search (retrieve_context)", value=True)
    use_expansion = st.checkbox("Use query expansion", value=True)
    similarity_threshold = st.slider("Similarity threshold", min_value=0.0, max_value=1.0, value=0.50, step=0.01)
    uploaded_files = st.file_uploader("Upload one or more txt/PDFs", type=["pdf", "txt"], accept_multiple_files=True)

    if uploaded_files and st.button("Add Index"):
        with st.spinner("Indexing documents..."):
            docs = []
            for file in uploaded_files:
                documents_dir = os.path.join(RAG_INDEX_DIR, "documents")
                os.makedirs(documents_dir, exist_ok=True)
                document_path = os.path.join(documents_dir, file.name)
                with open(document_path, "wb") as f:
                    f.write(file.getvalue())
                
                loader = PyPDFLoader(document_path) if file.name.endswith(".pdf") else TextLoader(document_path, autodetect_encoding=True)
                docs.extend(loader.load())

            text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=200)
            chunks = text_splitter.split_documents(docs)

            st.session_state.vector_store.add_documents(chunks)
            os.makedirs(RAG_INDEX_DIR, exist_ok=True)           
            st.session_state.vector_store.save_local(RAG_INDEX_PATH)
            st.success(f"{len(uploaded_files)} files indexed successfully!")

if st.session_state.model != selected_model or st.session_state.llm is None:
    # Initialize LLM if model changed 
    st.session_state.model = selected_model
    st.session_state.llm = ChatOllama(model=selected_model, base_url=OLLAMA_URL)
    logger.info(f"LLM initialized with model: {selected_model}")
    
llm = st.session_state.llm

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    st.chat_message(msg.type).write(msg.content)

if user_input := st.chat_input("Ask a question..."):
    st.chat_message("human").write(user_input)
    st.session_state.messages.append(HumanMessage(content=user_input))

    with st.spinner("Thinking..."):
        final_query = user_input
        if use_expansion:
            chain = build_query_expansion_chain(query_expansion_messages, llm)
            expanded_query = chain.invoke({"query": user_input})
            if expanded_query.strip() != user_input:
                final_query = expanded_query.strip()
                with st.chat_message("ai"):
                    st.markdown("🔍 **Expanded Query**")
                    st.markdown(final_query)

        response_container = st.chat_message("ai").empty()
        full_response = ""
        
        messages = rag_prompt_messages if use_context else simple_prompt_messages
        chain = build_rag_chain(messages, st.session_state.vector_store, similarity_threshold, llm)        
        for chunk in chain.stream({"query": final_query}):
            full_response += chunk
            response_container.markdown(full_response)
        
    st.session_state.messages.append(AIMessage(content=full_response))
