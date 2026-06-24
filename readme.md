# Local RAG Document Q&A Chatbot 🤖📄

A modern, local Retrieval-Augmented Generation (RAG) document Q&A chatbot built with **Streamlit**, **ChromaDB**, and **Ollama**. Upload your PDF documents and ask questions in natural language. The system retrieves relevant chunks of text from the PDF, feeds them into your local LLM, and highlights the source references with page numbers.

All data stays 100% local on your machine!

---

## 🌟 Features
- **Local PDF Processing**: Upload any PDF file and extract text dynamically.
- **Smart Chunking & Page Mapping**: Splits documents into manageable chunks while maintaining reference to the original page numbers.
- **Semantic Vector Storage**: Store document embeddings in a persistent local **ChromaDB** instance.
- **Local LLM & Embeddings**: Uses **Ollama** for both embedding generation (`nomic-embed-text`) and chat generation (e.g., `deepseek-r1:7b`, `llama3`).
- **Dynamic Model Selector**: Detects and displays installed models from your local Ollama instance automatically.
- **Chat History**: Remembers previous questions and answers in the session.
- **Source Chunk Highlighting**: Displays the exact retrieved text passages used to generate the answer, with matching keywords highlighted.

---

## 🛠️ Tech Stack
- **Frontend / UI**: [Streamlit](https://streamlit.io/)
- **Vector Database**: [ChromaDB](https://www.trychroma.com/)
- **Local LLM Orchestrator**: [Ollama](https://ollama.com/)
- **PDF Parser**: [PyPDF](https://pypi.org/project/pypdf/)
- **HTTP client**: [Requests](https://requests.readthedocs.io/)

---

## ⚙️ Setup & Installation

### 1. Prerequisites
- **Python 3.9+** installed on your machine.
- **Docker Desktop** installed (if running Ollama inside a container).

---

### 2. Start the Ollama Container
Since you are using Ollama in Docker, start your container.
You can run this command in your terminal or start the container via the Docker Desktop interface:

```bash
docker start ollama
```

---

### 3. Pull Required Models
To perform document Q&A, you will need an **embedding model** and a **chat model**. Run these inside the Ollama container:

```bash
# Pull the embedding model (highly recommended for RAG vector search)
docker exec ollama ollama pull nomic-embed-text

# Pull your preferred chat model (e.g., deepseek-r1:7b, llama3, or mistral)
docker exec ollama ollama pull deepseek-r1:7b
```

---

### 4. Install Python Dependencies
Install the required packages using `pip`:

```bash
pip install -r requirements.txt
```

*Note: The `requirements.txt` file installs `streamlit`, `chromadb`, `pypdf`, and `requests`.*

---

## 🚀 Running the Chatbot

Start the Streamlit application:

```bash
streamlit run app.py
```

This will spin up a local web server (usually at `http://localhost:8501`) and open the chatbot interface in your browser.

---

## 💡 How to Use
1. **Connect to Ollama**: The app automatically attempts to connect to Ollama at `http://localhost:11434`.
2. **Select Models**: In the sidebar, select the embedding model (`nomic-embed-text`) and the chat LLM (`deepseek-r1:7b` or another local model).
3. **Upload PDF**: Drag & drop your PDF file in the sidebar. The app will extract the text, split it into chunks, embed it, and store it in ChromaDB.
4. **Chat**: Ask questions in natural language.
5. **View Sources**: Expand the **"Retrieved Source Passages"** section under each answer to see the exact text chunks used, including their page numbers and highlighted keywords matching your query.

