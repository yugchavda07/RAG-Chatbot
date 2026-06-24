"""
Local RAG Document Q&A Chatbot - Streamlit Frontend
Orchestrates UI interaction, document upload, vector storage ingestion,
similarity retrieval, and local LLM chat completions with custom source highlighting.
"""

import sys
import os
import streamlit as st

# Ensure modular imports from current directory work correctly
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ollama_client import OllamaClient
from pdf_processor import PDFProcessor
from vector_store import VectorStoreManager

# --- PAGE SETUP & CUSTOM STYLING ---
st.set_page_config(
    page_title="Local RAG Chatbot",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Premium dark-theme optimized custom styles
st.markdown("""
<style>
    /* Remove upper top padding */
    .block-container {
        padding-top: 1.5rem !important;
        padding-bottom: 1.5rem !important;
    }

    /* Remove upper top padding in sidebar */
    [data-testid="stSidebarUserContent"] {
        padding-top: 0.2rem !important;
    }

    /* Chat Bubble styling */
    .chat-container {
        display: flex;
        flex-direction: column;
        gap: 15px;
        margin-bottom: 25px;
        max-height: 600px;
        overflow-y: auto;
        padding-right: 10px;
    }
    .user-bubble {
        align-self: flex-end;
        background-color: #2b5c8f;
        color: #ffffff;
        padding: 12px 18px;
        border-radius: 18px 18px 2px 18px;
        max-width: 80%;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        word-wrap: break-word;
        font-family: inherit;
    }
    .assistant-bubble {
        align-self: flex-start;
        background-color: #1e1e24;
        color: #e0e0e0;
        padding: 12px 18px;
        border-radius: 18px 18px 18px 2px;
        max-width: 80%;
        box-shadow: 0 2px 5px rgba(0,0,0,0.15);
        border: 1px solid #2d2d34;
        word-wrap: break-word;
        font-family: inherit;
    }
    .sender-label {
        font-size: 0.8rem;
        color: #868e96;
        margin-bottom: 4px;
    }
    .user-wrapper {
        display: flex;
        flex-direction: column;
        align-items: flex-end;
    }
    .assistant-wrapper {
        display: flex;
        flex-direction: column;
        align-items: flex-start;
    }
    
    /* Document chunk references styling */
    .source-card {
        background-color: #17181c;
        border-left: 4px solid #4dabf7;
        padding: 12px;
        margin-bottom: 12px;
        border-radius: 4px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.2);
    }
    .source-header {
        font-weight: bold;
        font-size: 0.85rem;
        color: #4dabf7;
        margin-bottom: 6px;
        display: flex;
        justify-content: space-between;
    }
    .source-text {
        font-size: 0.9rem;
        color: #ced4da;
        line-height: 1.5;
    }
</style>
""", unsafe_allow_html=True)

# --- INITIALIZE CORE SERVICES ---
# Persistence directory inside the subdirectory to avoid path mixups
PERSIST_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chroma_db")
vector_manager = VectorStoreManager(persist_dir=PERSIST_DIR)

# --- REUSABLE HELPER FUNCTIONS ---

def highlight_keywords(text: str, query: str) -> str:
    """
    Highlights terms in the text matching keywords from the query.
    Uses clean yellow highlights visible in both dark and light modes.
    """
    import re
    # Filter stopwords to only highlight content-carrying words
    stopwords = {
        "what", "is", "the", "a", "an", "of", "and", "in", "to", "for", "on", "with", 
        "at", "by", "from", "how", "why", "where", "who", "which", "are", "about", 
        "it", "this", "that", "these", "those", "you", "your", "i", "we", "they", 
        "he", "she", "me", "us", "them", "him", "her", "can", "could", "would", 
        "should", "will", "do", "does", "did", "have", "has", "had", "or", "but",
        "as", "if", "then", "else", "when", "there", "their", "so", "than"
    }
    
    # Tokenize query into words, filtering out punctuation
    query_words = re.findall(r'\b\w+\b', query.lower())
    keywords = [w for w in query_words if w not in stopwords and len(w) > 2]
    
    # Fallback to general words if query is entirely stopwords
    if not keywords:
        keywords = [w for w in query_words if len(w) > 2]
        
    if not keywords:
        return text
        
    # Sort keywords by length in descending order to avoid matching nested words incorrectly
    keywords = sorted(list(set(keywords)), key=len, reverse=True)
    
    # Compile regex pattern to match whole words case-insensitively
    pattern = re.compile(rf'\b({"|".join(map(re.escape, keywords))})\b', re.IGNORECASE)
    
    def replace_match(match):
        word = match.group(0)
        return f'<mark style="background-color: #ffd43b; color: #212529; font-weight: 600; border-radius: 3px; padding: 1px 3px;">{word}</mark>'
        
    return pattern.sub(replace_match, text)

# --- SESSION STATE INITIALIZATION ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "current_file" not in st.session_state:
    st.session_state.current_file = None
if "collection_name" not in st.session_state:
    st.session_state.collection_name = None
if "last_sources" not in st.session_state:
    st.session_state.last_sources = []
if "last_query" not in st.session_state:
    st.session_state.last_query = ""

# --- SIDEBAR - SERVICE SETUP & DOCUMENT INGESTION ---
with st.sidebar:
    st.title("⚙️ Settings & Uploads")
    
    # 1. Ollama Connection Status & Base URL
    st.markdown("### 🔌 Ollama Connection")
    ollama_url = st.text_input("Ollama Endpoint URL", value="http://localhost:11434")
    
    # Instantiate Ollama client
    ollama_client = OllamaClient(base_url=ollama_url)
    is_connected = ollama_client.check_connection()
    
    if is_connected:
        st.success("Ollama Status: Connected 🟢")
    else:
        st.error("Ollama Status: Disconnected 🔴")
        st.warning("Please start your Ollama Docker container:\n`docker start ollama`")
    
    st.markdown("---")
    
    # 2. Model Selection dropdowns (conditional on connection)
    available_models = []
    if is_connected:
        available_models = ollama_client.list_models()
        
    st.markdown("### 🤖 Model Configurations")

    # Known keywords that identify embedding-only models (cannot do chat)
    EMBED_KEYWORDS = ["embed", "minilm", "bge-", "gte-", "e5-", "nomic-embed"]

    def is_embed_model(name: str) -> bool:
        """Returns True if the model name matches a known embedding-only pattern."""
        return any(kw in name.lower() for kw in EMBED_KEYWORDS)

    # Separate models into two groups
    embed_models = [m for m in available_models if is_embed_model(m)]
    chat_models  = [m for m in available_models if not is_embed_model(m)]

    # Fallbacks if no models detected yet
    if not embed_models:
        embed_models = ["nomic-embed-text"]
    if not chat_models:
        chat_models = ["deepseek-r1:7b"]

    # Embedding Model selection — only shows embedding-capable models
    embedding_model = st.selectbox(
        "Select Embedding Model",
        options=embed_models,
        index=0
    )

    # Chat LLM Selection — only shows generative / chat-capable models
    default_chat = "deepseek-r1:7b"
    chat_default_idx = chat_models.index(default_chat) if default_chat in chat_models else 0
    chat_model = st.selectbox(
        "Select Chat Model (LLM)",
        options=chat_models,
        index=chat_default_idx
    )
    
    st.markdown("---")
    
    # 3. Document Uploader
    st.markdown("### 📁 Document Upload")
    uploaded_file = st.file_uploader("Upload a PDF document", type=["pdf"])
    
    if uploaded_file is not None and is_connected:
        # Check if it's a new file
        if st.session_state.current_file != uploaded_file.name:
            st.session_state.current_file = uploaded_file.name
            
            # Generate sanitized collection name for ChromaDB
            collection_name = vector_manager.sanitize_collection_name(uploaded_file.name)
            st.session_state.collection_name = collection_name
            
            # Check if this PDF is already processed and stored in ChromaDB
            if vector_manager.collection_exists_and_has_data(collection_name):
                st.info(f"Loaded existing data for {uploaded_file.name} 📂")
            else:
                with st.spinner("Parsing PDF & extracting text by page..."):
                    pages_data = PDFProcessor.extract_text_by_page(uploaded_file)
                    chunks = PDFProcessor.chunk_pages(pages_data, chunk_size=800, chunk_overlap=150)
                
                if chunks:
                    with st.spinner("Generating embeddings & saving to ChromaDB..."):
                        # Extract raw texts for embedding
                        raw_texts = [c["text"] for c in chunks]
                        try:
                            # Batch embed texts
                            embeddings = ollama_client.get_embeddings_batch(raw_texts, embedding_model)
                            # Persist in ChromaDB
                            vector_manager.add_documents(collection_name, chunks, embeddings)
                            st.success(f"Ingested {len(chunks)} text chunks successfully! 🚀")
                        except Exception as e:
                            st.error(f"Failed embedding: {e}")
                else:
                    st.error("No extractable text found in this PDF.")
                    
    elif uploaded_file is None:
        # Clean current document states if file removed
        st.session_state.current_file = None
        st.session_state.collection_name = None
        st.session_state.last_sources = []
        st.session_state.last_query = ""

    st.markdown("---")
    
    # 4. Action Buttons (Maintenance)
    st.markdown("### 🧹 Utilities")
    col_clear, col_reset = st.columns(2)
    with col_clear:
        if st.button("Clear Chat 💬"):
            st.session_state.messages = []
            st.session_state.last_sources = []
            st.session_state.last_query = ""
            st.experimental_rerun()
            
    with col_reset:
        if st.button("Reset DB 🗑️"):
            if st.session_state.collection_name:
                vector_manager.delete_collection(st.session_state.collection_name)
                st.session_state.current_file = None
                st.session_state.collection_name = None
                st.session_state.last_sources = []
                st.session_state.last_query = ""
                st.success("Vector DB reset completed!")
                st.experimental_rerun()
            else:
                st.info("No active document loaded to delete.")

# --- MAIN UI LAYOUT ---

# Header block
st.markdown("""
<div style="padding: 1rem 0; margin-bottom: 2rem; border-bottom: 1px solid #2d2d34;">
    <h1 style="color: #4dabf7; margin: 0; font-size: 2.25rem;">🧠 Local RAG Document Chatbot</h1>
    <p style="color: #868e96; margin: 5px 0 0 0; font-size: 1.05rem;">
        Ask questions about your uploaded PDF. Everything is processed locally via Ollama & ChromaDB.
    </p>
</div>
""", unsafe_allow_html=True)

# Split screen layout: Left for Chat, Right for context sources
col_chat, col_sources = st.columns([1.8, 1.2])

# Left column: Chat History & Input
with col_chat:
    st.subheader("Conversation")
    
    # Display chat logs using custom bubble structure
    if st.session_state.messages:
        chat_html = '<div class="chat-container">'
        for idx, msg in enumerate(st.session_state.messages):
            if msg["role"] == "user":
                chat_html += f"""
<div class="user-wrapper">
<span class="sender-label">You</span>
<div class="user-bubble">{msg['content']}</div>
</div>
"""
            else:
                chat_html += f"""
<div class="assistant-wrapper">
<span class="sender-label">Assistant ({chat_model})</span>
<div class="assistant-bubble">{msg['content']}</div>
</div>
"""
        chat_html += '</div>'
        st.markdown(chat_html, unsafe_allow_html=True)
    else:
        st.info("👋 Upload a PDF and type your question below to start the conversation!")

    # Chat Input form (Compatible with Streamlit 1.12.0)
    # Allows enter key submissions
    st.markdown("---")
    
    # Disable input if not connected to Ollama or no PDF uploaded
    input_disabled = not is_connected or st.session_state.collection_name is None
    
    placeholder_text = "Ask a question about the document..."
    if not is_connected:
        placeholder_text = "⚠️ Please connect and start Ollama..."
    elif st.session_state.collection_name is None:
        placeholder_text = "📂 Please upload a PDF file from the sidebar to chat..."

    with st.form(key="chat_input_form", clear_on_submit=True):
        user_query = st.text_input(
            label="Ask a question:",
            placeholder=placeholder_text,
            disabled=input_disabled
        )
        submit_query = st.form_submit_button("Send Question 🚀")

    # Process Query on Submit
    if submit_query and user_query.strip():
        # Append user message
        st.session_state.messages.append({"role": "user", "content": user_query})
        st.session_state.last_query = user_query
        
        # Ingest loading block
        with st.spinner("Searching document & generating answer..."):
            try:
                # 1. Retrieve query embeddings
                query_vector = ollama_client.get_embedding(user_query, embedding_model)
                
                # 2. Semantic Search in ChromaDB
                retrieved_chunks = vector_manager.query_similarity(
                    collection_name=st.session_state.collection_name,
                    query_embedding=query_vector,
                    top_k=4
                )
                
                # Cache sources to display in the side panel
                st.session_state.last_sources = retrieved_chunks
                
                # 3. Build Prompt Context
                context_str = "\n\n".join([
                    f"[Source: Page {chunk['page_number']}]\n{chunk['text']}"
                    for chunk in retrieved_chunks
                ])
                
                system_prompt = (
                    "You are a professional document Q&A assistant. Use the following context "
                    "extracted from a document to answer the user's question accurately.\n"
                    "If the answer cannot be found in the context, state that you cannot find the answer "
                    "in the document. Keep your answers concise, accurate, and professional.\n\n"
                    f"CONTEXT:\n{context_str}"
                )
                
                # 4. Construct message history payload for Ollama chat endpoint
                # Includes system prompt containing current context + full session message history
                api_messages = [{"role": "system", "content": system_prompt}]
                
                # Append last 6 conversation exchanges to maintain context without overloading window
                for past_msg in st.session_state.messages[:-1][-6:]:
                    api_messages.append({"role": past_msg["role"], "content": past_msg["content"]})
                    
                # Append current user prompt
                api_messages.append({"role": "user", "content": user_query})
                
                # 5. Call LLM (non-streaming placeholder for response, or we stream to placeholder)
                # Since we want a nice interface, let's stream the response to the screen in real-time
                response_placeholder = st.empty()
                full_response = ""
                
                # We can stream tokens directly to our custom bubble container
                for token in ollama_client.chat_completion_stream(chat_model, api_messages):
                    full_response += token
                    # Style response dynamically while it streams
                    response_placeholder.markdown(f"""
<div class="assistant-wrapper">
<span class="sender-label">Assistant ({chat_model}) [Typing...]</span>
<div class="assistant-bubble">{full_response}</div>
</div>
""", unsafe_allow_html=True)
                
                # Add LLM response to messages state
                st.session_state.messages.append({"role": "assistant", "content": full_response})
                
                # Trigger screen refresh to draw standard message history
                st.experimental_rerun()
                
            except Exception as e:
                st.error(f"Error occurred during search/generation: {e}")

# Right column: Context / Source references Panel
with col_sources:
    st.subheader("📚 Source References")
    
    if st.session_state.collection_name is None:
        st.info("Upload a PDF to view semantic context chunks here.")
    elif not st.session_state.last_sources:
        st.info("Ask a question to see the retrieved page passages and matching highlights.")
    else:
        st.markdown(f"**Retrieved Passages for:** *\"{st.session_state.last_query}\"*")
        
        # Display each retrieved source chunk with custom highlighting
        for i, chunk in enumerate(st.session_state.last_sources):
            # Highlight terms in the text based on the query keywords
            highlighted_text = highlight_keywords(chunk["text"], st.session_state.last_query)
            distance_score = chunk.get("distance", 0.0)
            
            st.markdown(f"""
<div class="source-card">
<div class="source-header">
<span>Page {chunk['page_number']}</span>
<span>Relevance Distance: {distance_score:.4f}</span>
</div>
<div class="source-text">{highlighted_text}</div>
</div>
""", unsafe_allow_html=True)
