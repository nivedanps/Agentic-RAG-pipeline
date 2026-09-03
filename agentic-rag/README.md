# Local Agentic RAG AI Assistant

A complete Local Agentic Retrieval-Augmented Generation (RAG) system with genuine autonomous decision-making.

## 🌟 What is Agentic RAG?

**Traditional RAG** follows a simple pipeline:
User Question → Vector Search → Pass results to LLM → Answer

**Agentic RAG** introduces an autonomous AI agent that manages the process:
User Question → Agent Analyzes Intent → Agent Decides whether to use search tools → Evaluates Results → Rewrites Query if needed → Retry Search → Generate Answer.

### Why this matters:
1. **No unnecessary retrieval**: If you say "Hello", the agent answers directly instead of searching documents for "Hello".
2. **Quality Control**: If a search returns bad results, the agent realizes it, rewrites the query, and searches again.
3. **No Hallucinations**: If the information simply isn't in the documents, the agent can gracefully say so instead of making things up.

## 🏗️ Architecture & Technologies Used

The entire system runs **100% locally**. No cloud API keys, no data leaves your machine.

- **Backend**: Python 3.11, FastAPI
- **LLM**: Ollama (`llama3.2` default)
- **Agent Workflow**: LangGraph, LangChain
- **Embeddings**: HuggingFace (`sentence-transformers/all-MiniLM-L6-v2`)
- **Vector Database**: ChromaDB (Persistent local storage)
- **Frontend**: Modern Vanilla JS + HTML + CSS (Glassmorphism design)

## 🚀 Installation & Setup

### 1. Install Ollama
Download and install Ollama from [ollama.com](https://ollama.com/).

### 2. Pull the required model
Open a terminal and run:
```bash
ollama pull llama3.2
```

### 3. Setup Python Environment
Ensure you have Python 3.11+ installed.
```bash
cd agentic-rag
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 4. Configuration (Optional)
Copy `.env.example` to `.env` if you want to tweak settings (like chunk size or model names).
```bash
cp .env.example .env
```

## 🏃‍♂️ Running the Application

### Start the Backend
```bash
# Ensure your virtual environment is activated
uvicorn backend.main:app --reload --port 8000
```

### Start the Frontend
Simply open the `frontend/index.html` file in your web browser!
(You can double click it, or use a tool like Live Server).

## 💡 How to Use

1. **Upload Documents**: Use the left panel to upload PDF, TXT, or DOCX files.
2. **Ask Questions**: Use the chat panel to ask questions.
3. **Watch the Agent**: Look at the right panel ("Agent Activity") to see exactly how the agent is reasoning:
   - Is it answering directly?
   - Is it searching documents?
   - Did it have to retry the search with a new query?

### 🎯 Example Questions to Try

1. **Direct Answer (No search)**: "Hello! What can you do?"
2. **Direct Answer (General Knowledge)**: "What is the capital of France?"
3. **Fact Retrieval**: "According to the annual report, what was the revenue in 2024?"
4. **Summarization**: "Summarize the key points from the uploaded document."
5. **Multi-step Reasoning**: "Compare the 2023 and 2024 revenue margins."

## 🧠 Understanding the Agent Workflow (LangGraph)

The core logic is in `backend/agent.py`. The agent operates as a state machine:

1. **Analyze**: LLM decides if retrieval is needed based on the question.
2. **Retrieve**: If needed, calls the ChromaDB search tool.
3. **Evaluate**: LLM reviews the retrieved chunks. Are they sufficient?
4. **Rewrite**: If insufficient, LLM writes a better search query and loops back to Retrieve (up to 3 times).
5. **Generate**: Finally, answers using only the verified retrieved context, citing sources.

## 🛠️ Tools Explained

The agent has access to specific tools (`backend/agent.py`):
- `search_documents`: The primary tool. Performs semantic search across all indexed documents.
- `list_documents`: Lets the agent check what files are available.
- `search_specific_document`: Lets the agent target a single file.
- `document_summary`: Pulls representative chunks from across a document to provide a high-level overview.

## ⚠️ Current Limitations

- Extremely large documents (>1000 pages) may take significant time to embed using the local CPU-bound embedding model.
- The default `llama3.2` model is fast but small; for highly complex reasoning, a larger model like `llama3.1:8b` might yield better logic (configurable in `.env`).
- OCR for image-based PDFs is not implemented.

---
Built as a complete Local Agentic RAG Learning Project.
