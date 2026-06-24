# Step-by-Step Implementation Guide 🛠️🧠

This document explains exactly how we built this local RAG Document Q&A chatbot, step-by-step, in the simplest possible terms.

---

## 🗺️ What is RAG? (The Textbook Analogy)
Imagine you have an open-book exam.
* **The PDF** is the textbook.
* **The LLM (DeepSeek)** is the student.
* **The Vector DB (ChromaDB)** is the book's index.

Instead of making the student read the whole textbook (which is too long), when you ask a question, we look up the keyword in the index, find the exact pages, show them to the student, and ask them to write the answer based *only* on those pages. This is **Retrieval-Augmented Generation (RAG)**!

---

## 🚀 How to Run & Use the Application

### 1. How to Run
Make sure you are in your command terminal and follow these commands:
       
1. **Start Ollama** (starts the Docker container):
   ```bash
   docker start ollama
   ```
2. **Install Python Packages** (installs Streamlit, ChromaDB, PyPDF, and Requests):
   ```bash
   pip install -r RAG-Chatbot/requirements.txt
   ```
3. **Launch the App**:
   ```bash
   cd RAG-Chatbot
   streamlit run app.py
   ```

---

### 2. How to Use (Step-by-Step in the UI)
Once the application window opens in your web browser (`http://localhost:8501`):

1. **Verify Connection**: Look at the sidebar. You should see a green **"Ollama Status: Connected 🟢"** box.
2. **Select Models**: 
   * Under **Select Embedding Model**, select `nomic-embed-text`.
   * Under **Select Chat Model (LLM)**, select `deepseek-r1:7b` (or whichever local model you prefer).
3. **Upload PDF**: Click **"Browse files"** in the sidebar and select your PDF file.
   * The app automatically processes and saves it to ChromaDB.
   * A success message will appear showing how many chunks were ingested.
4. **Chat**: Type a natural language question in the box at the bottom and click **"Send Question 🚀"**.
5. **Check Sources**: Look at the right panel **"Source References"** to see the exact text snippets pulled from the PDF along with their page numbers, with key matching words highlighted in yellow!

---

## 🚪 Step 1: Set Up Docker & Ollama (The Brain)
We run our AI models locally inside a Docker container using Ollama.

1. **Why Docker?** 
   Docker keeps Ollama isolated in its own virtual container so it doesn't conflict with other programs on your computer.
2. **Start the Container**:
   We start the Ollama container so its service is active at `http://localhost:11434`:
   ```bash
   docker start ollama
   ```
3. **What models do we need?**
   * **Chat Model (`deepseek-r1:7b`)**: The language model that understands prompts and writes answers.
   * **Embedding Model (`nomic-embed-text`)**: A specialized model that converts words into numbers (vectors) representing their meaning.
4. **Pull models inside Docker**:
   We tell Docker to execute the pull command inside the running Ollama container:
   ```bash
   docker exec ollama ollama pull nomic-embed-text
   docker exec ollama ollama pull deepseek-r1:7b
   ```

---

## 🐍 Step 2: Install Python Libraries (The Tools)
We need a few Python packages to make this work:
```bash
pip install streamlit chromadb pypdf requests altair<5
```
* **`streamlit`**: Builds the web interface using pure Python.
* **`chromadb`**: The vector database to store our parsed text.
* **`pypdf`**: Extracts text from PDF files.
* **`requests`**: Communicates with Ollama's HTTP API.

> **⚠️ Version Note**: Because your Python version is **3.9.7**, we installed **Streamlit 1.12.0** and downgraded **Altair to 4.2.2** to prevent import conflicts (newer Streamlit versions blacklist Python 3.9.7).

---

## 📄 Step 3: Read and Chunk the PDF (`pdf_processor.py`)
1. **Reading**: We use `pypdf` to read the uploaded PDF page-by-page.
2. **Chunking**: An LLM has a "context window" limit (it can only read so much text at once). We split the extracted text into small chunks of **800 characters**.
3. **Overlap**: We make chunks overlap by **150 characters**. This ensures we don't accidentally cut a sentence or paragraph in half and lose the meaning.
4. **Page Mapping**: We save the **page number** with each chunk. This is crucial for showing references to the user!

---

## 🗄️ Step 4: Save to the Vector DB (`vector_store.py`)
1. **What is an Embedding?**
   An embedding is a list of decimal numbers (a vector) that represents the *meaning* of a text chunk. Similar meanings will have similar numbers.
2. **Why ChromaDB & SQLite?**
   * **ChromaDB** is a simple, lightweight vector database.
   * By default, it uses **SQLite** (a self-contained SQL database engine) to save data to a file on your disk (`./chroma_db`).
   * This is perfect because you don't need to run a complex, external database server. ChromaDB does everything inside your Python script!
3. **Ingestion**:
   * We convert all PDF chunks to embeddings using `nomic-embed-text` via Ollama.
   * We save the text chunk, its embedding, and the page number in ChromaDB.

---

## 💬 Step 5: Talking to the LLM (`ollama_client.py`)
1. **Semantic Search**:
   * When you type a question, we convert your question into an embedding.
   * We search ChromaDB to find the **top 4 chunks** most similar to your question.
2. **Constructing the Prompt**:
   We package the retrieved PDF chunks and send them to the chat model:
   ```text
   Context: [Retrieved PDF Page 2 & Page 5 text]
   Question: [User's Question]
   Answer:
   ```
3. **Streaming**:
   We stream the response from Ollama token-by-token (word-by-word) so the user doesn't have to wait for the whole answer to generate before seeing it.

---

## 🎨 Step 6: Create the User Interface (`app.py`)
1. **Double Column Layout**:
   * **Left Side**: Chat area with custom styled HTML/CSS bubbles.
   * **Right Side**: Source references panel showing where the info came from.
2. **Keyword Highlighting**:
   * We extract keywords from your question (ignoring stop words like "the", "is", "a").
   * We search for those keywords in the retrieved PDF chunks.
   * We wrap those keywords in HTML `<mark>` tags to highlight them in yellow.

---

## 📁 What is `__pycache__`?
When you run a Python script, you might notice a folder called `__pycache__` appearing.
* **Why does it make this?**
  Python compiles your human-readable `.py` files into machine-readable **bytecode** (stored as `.pyc` files).
* **What does it do?**
  Next time you run the app, Python loads the pre-compiled `.pyc` files instead of compiling them from scratch. This makes imports and startups **much faster**!
* **Can you delete it?**
  Yes, it is completely safe to delete. Python will simply recreate it the next time you run the code. You should add it to your `.gitignore` file.
