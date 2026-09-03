import os
import tempfile
from pathlib import Path
# pyrefly: ignore [missing-import]
import pytest
# pyrefly: ignore [missing-import]
from fastapi.testclient import TestClient
from langchain_core.documents import Document

from backend.main import app
from backend.schemas import create_initial_state, add_trace
from backend.agent import get_agent_graph, format_conversation_context, document_summary
from backend.rag import load_txt, load_document, split_documents

client = TestClient(app)

def test_initial_state():
    state = create_initial_state("Hello, world!")
    assert state["user_question"] == "Hello, world!"
    assert state["requires_retrieval"] is False
    assert state["retrieval_attempts"] == 0
    assert len(state["agent_trace"]) == 0

def test_add_trace():
    state = create_initial_state("Test")
    add_trace(state, "test_step", "Testing...")
    assert len(state["agent_trace"]) == 1
    assert state["agent_trace"][0]["step"] == "test_step"
    assert state["agent_trace"][0]["message"] == "Testing..."
    assert "timestamp" in state["agent_trace"][0]

def test_agent_graph_caching():
    graph1 = get_agent_graph()
    graph2 = get_agent_graph()
    assert graph1 is graph2

def test_format_conversation_context():
    history = [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello"}
    ]
    context = format_conversation_context(history)
    assert "USER: Hi" in context
    assert "ASSISTANT: Hello" in context

def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "ollama_available" in data
    assert "chroma_available" in data

def test_documents_endpoint_empty():
    response = client.get("/documents")
    assert response.status_code == 200
    data = response.json()
    assert "documents" in data
    assert "total_count" in data

def test_chat_validation_error():
    response = client.post("/chat", json={"question": ""})
    assert response.status_code == 422

    response = client.post("/chat", json={})
    assert response.status_code == 422

def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "running"
    assert "app" in data

def test_frontend_app_endpoint():
    response = client.get("/app/")
    assert response.status_code == 200
    assert "text/html" in response.headers.get("content-type", "")
    assert "Agentic RAG" in response.text

def test_txt_loader():
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write("This is a test document.")
        temp_path = f.name
    try:
        docs = load_txt(Path(temp_path))
        assert len(docs) == 1
        assert docs[0].page_content == "This is a test document."
        assert docs[0].metadata["source"] == Path(temp_path).name
        assert docs[0].metadata["file_type"] == "txt"
    finally:
        os.unlink(temp_path)

def test_universal_loader_txt():
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write("Universal loader test.")
        temp_path = f.name
    try:
        docs = load_document(Path(temp_path))
        assert len(docs) == 1
        assert docs[0].page_content == "Universal loader test."
    finally:
        os.unlink(temp_path)

def test_unsupported_file_type():
    with pytest.raises(ValueError, match="Unsupported file type"):
        load_document(Path("test.csv"))

def test_text_splitter():
    long_text = "Word. " * 300
    doc = Document(page_content=long_text, metadata={"source": "test.txt"})
    chunks = split_documents([doc], chunk_size=100, chunk_overlap=20)
    assert len(chunks) > 1
    for i, chunk in enumerate(chunks):
        assert chunk.metadata["source"] == "test.txt"
        assert chunk.metadata["chunk_index"] == i
        assert len(chunk.page_content) <= 100

def test_text_splitter_zero_overlap():
    long_text = "Word. " * 50
    doc = Document(page_content=long_text, metadata={"source": "zero.txt"})
    chunks = split_documents([doc], chunk_size=50, chunk_overlap=0)
    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk.page_content) <= 50

def test_summary_tool_empty_db():
    result = document_summary("non_existent_file.txt")
    assert result["found"] is False
    assert result["chunk_count"] == 0
    assert "No content found" in result["context"]

def test_txt_loader_empty_error():
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write("   \n\n  ")
        temp_path = f.name
    try:
        with pytest.raises(ValueError, match="empty"):
            load_txt(Path(temp_path))
    finally:
        os.unlink(temp_path)
