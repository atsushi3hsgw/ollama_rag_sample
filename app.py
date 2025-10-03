import streamlit as st
from sentence_transformers import SentenceTransformer
import faiss
import pickle
import os
import yaml
from dotenv import load_dotenv

from rag_loader import load_pdf_text, load_text_file, split_text, split_text_with_window
from ollama_client import query_ollama, get_available_models

@st.cache_data
def load_prompt_templates():
    """
    Loading prompt templates for each model
    """
    with open("prompts.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
    
def build_prompt(query: str, context: str, model: str, templates: dict) -> str:
    """
    Build prompt according to the prompt template for each model
    """
    matched_key = next((k for k in templates if k in model), None)
    if not matched_key:
        return f"Question: {query}"

    template_set = templates[matched_key]
    if context.strip():
        template = template_set.get("template", "")
        return template.replace("{{query}}", query).replace("{{context}}", context.strip())
    else:
        template = template_set.get("empty_context", "")
        return template.replace("{{query}}", query)

load_dotenv()
rag_index_dir = os.path.join(".", "./rag_index_dir")
rag_index_path = os.path.join(rag_index_dir, "rag_index.pkl")

# Load embedding model
if "embedder" not in st.session_state:
    st.session_state.embedder = SentenceTransformer(
        os.getenv("EMBEDDING_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"))

# Load index if exists
if "index" not in st.session_state:
    if os.path.exists(rag_index_path):
        with open(rag_index_path, "rb") as f:
            index, documents = pickle.load(f)
            st.session_state.index = index
            st.session_state.documents = documents
    else:
        st.session_state.index = None
        st.session_state.documents = []
        
with st.sidebar:
    st.header("Configs")
    model_list = get_available_models()
    selected_model = st.selectbox("model", model_list)
    st.session_state.model = selected_model
    
    use_context = st.checkbox("RAG search（retrieve_context）", value=True)
    st.session_state.use_context = use_context
    
    use_expansion = st.checkbox("Use query expansion", value=True)
    st.session_state.use_expansion = use_expansion
    
    threshold = st.slider("Similarity threshold", min_value=0.0, max_value=1.0, value=0.50, step=0.01)
    st.session_state.threshold = threshold

    # upload files
    uploaded_files = st.file_uploader("Upload one or more txt/PDFs", type=["pdf", "txt"], accept_multiple_files=True)

    if uploaded_files:
        if st.button("Add Index"):
            new_chunks = []
            for file in uploaded_files:
                os.makedirs(rag_index_dir, exist_ok=True)
                upload_path = os.path.join(rag_index_dir, file.name)
                with open(upload_path, "wb") as f:
                    f.write(file.read())
    
                if file.name.endswith(".pdf"):
                    raw_text = load_pdf_text(upload_path)
                elif file.name.endswith(".txt"):
                    raw_text = load_text_file(upload_path)
                else:
                    continue
    
                chunks = split_text_with_window(raw_text)
                new_chunks.extend(chunks)

            new_embeddings = st.session_state.embedder.encode(new_chunks, normalize_embeddings=True)
    
            if st.session_state.index is None:
                # st.session_state.index = faiss.IndexFlatL2(new_embeddings.shape[1])
                st.session_state.index = faiss.IndexFlatIP(new_embeddings.shape[1])
                st.session_state.index.add(new_embeddings)
                st.session_state.documents = new_chunks
            else:
                st.session_state.index.add(new_embeddings)
                st.session_state.documents.extend(new_chunks)
            
            # Save index
            os.makedirs(rag_index_dir, exist_ok=True)
            with open(rag_index_path, "wb") as f:
                pickle.dump((st.session_state.index, st.session_state.documents), f)
                st.success(f"{len(uploaded_files)} files uploaded and indexed successfully!")

def retrieve_context(query: str, top_k: int = 5, threshold: float = 0.50) -> str:
    """
    Find documents that is highly relevant to the query
    """
    
    if st.session_state.index is None:
        return ""
    query_vec = st.session_state.embedder.encode([query], normalize_embeddings=True)
    scores, indices = st.session_state.index.search(query_vec, top_k)
    # results = [st.session_state.documents[i] for i in indices[0]]
    results = []
    for score, idx in zip(scores[0], indices[0]):
        doc = st.session_state.documents[idx]
        print (f"Score:{score} Index:{idx} Doc:{doc}")  # Debugging line
        if score > threshold:
            if isinstance(doc, dict):
                results.append(f"[{doc['source']}] {doc['text']}")
            else:
                results.append(doc)
    if not results:
        return ""
    
    return "\n".join(results)

def generate_rag_response(user_query: str) -> str:
    templates = load_prompt_templates()
    
    if st.session_state.use_context:
        context = retrieve_context(user_query, threshold = st.session_state.threshold)
    else:
        context = ""
        
    prompt = build_prompt(user_query, context, st.session_state.model, templates)
    print("Prompt for LLM:\n", prompt)  # Debugging line
    return query_ollama(prompt, model=st.session_state.model)

def expand_query(query: str, model: str, templates: dict) -> str:
    key = next((k for k in templates if k in model), None)
    if not key:
        return query
    prompt = templates[key].get("query_expansion", "").replace("{{query}}", query)
    return query_ollama(prompt, model=model)

# Chat UI
st.title("RAG Chatbot with File Upload")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

user_input = st.chat_input("Ask a question...")

if user_input:
    st.chat_message("user").write(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.spinner("Thinking..."):
        if st.session_state.use_expansion:
            templates = load_prompt_templates()
            expanded_query = expand_query(user_input, st.session_state.model, templates)
            if expanded_query != user_input:
                with st.chat_message("assistant"):
                    st.markdown("🔍 **expanded query**")
                    st.markdown(expanded_query)
            user_input = expanded_query
            
        response = generate_rag_response(user_input).replace("\\n", "\n")
        st.chat_message("assistant").markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})
