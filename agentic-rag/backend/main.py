"""
main.py — FastAPI application entry point.

This file:
1. Creates the FastAPI app
2. Configures CORS (for frontend communication)
3. Defines all API endpoints
4. Checks Ollama availability on startup
5. Provides health monitoring

Run with: uvicorn backend.main:app --reload --port 8000
"""

import logging
import httpx
from pathlib import Path
from datetime import datetime
from contextlib import asynccontextmanager

# pyrefly: ignore [missing-import]
from fastapi import FastAPI, UploadFile, File, HTTPException
# pyrefly: ignore [missing-import]
from fastapi.middleware.cors import CORSMiddleware
# pyrefly: ignore [missing-import]
from fastapi.staticfiles import StaticFiles
# pyrefly: ignore [missing-import]
from fastapi.responses import FileResponse, JSONResponse

from backend.config import settings, setup_logging
from backend.schemas import (
    ChatRequest, ChatResponse,
    DocumentListResponse, DocumentInfo,
    UploadResponse, DeleteResponse,
    SearchRequest, SearchResponse, SourceCitation,
    HealthResponse
)
from backend.rag import (
    ingest_document, delete_document, get_all_documents,
    get_vector_store, retrieve
)
from backend.agent import process_chat

setup_logging()
logger = logging.getLogger("agentic_rag.main")

async def check_ollama() -> tuple[bool, bool, list]:
    """
    Check if Ollama is running and the configured model is available.
    
    Returns:
        (ollama_available, model_available, issues_list)
    """
    issues = []
    setup_instructions = []

    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{settings.ollama_base_url}/api/tags")
            if response.status_code == 200:
                ollama_available = True
                
                data = response.json()
                available_models = [m["name"] for m in data.get("models", [])]
                
                model_name = settings.ollama_model
                model_available = any(
                    m == model_name or m.startswith(model_name + ":") or m.startswith(model_name.split(":")[0])
                    for m in available_models
                )

                if not model_available:
                    issues.append(
                        f"Model '{settings.ollama_model}' not found in Ollama. "
                        f"Available: {', '.join(available_models) if available_models else 'none'}"
                    )
                    setup_instructions.append(
                        f"Run: ollama pull {settings.ollama_model}"
                    )
            else:
                ollama_available = False
                model_available = False
                issues.append("Ollama server returned an unexpected response")
    except Exception as e:
        ollama_available = False
        model_available = False
        issues.append(
            "Ollama is not running or not reachable at "
            f"{settings.ollama_base_url}. Error: {str(e)[:100]}"
        )
        setup_instructions.append("Start Ollama: ollama serve")
        setup_instructions.append(f"Pull model: ollama pull {settings.ollama_model}")

    return ollama_available, model_available, issues

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle application startup and shutdown."""
    logger.info("=" * 60)
    logger.info("  Agentic RAG AI Assistant — Starting Up")
    logger.info("=" * 60)
    logger.info(f"  Ollama URL:   {settings.ollama_base_url}")
    logger.info(f"  Ollama Model: {settings.ollama_model}")
    logger.info(f"  Embeddings:   {settings.embedding_model}")
    logger.info(f"  ChromaDB:     {settings.chroma_persist_directory}")
    logger.info("=" * 60)

    settings.data_dir.mkdir(parents=True, exist_ok=True)
    settings.chroma_dir.mkdir(parents=True, exist_ok=True)

    ollama_ok, model_ok, issues = await check_ollama()

    if not ollama_ok:
        logger.warning("⚠️  Ollama is NOT available! The AI agent will not function.")
        for issue in issues:
            logger.warning(f"   Issue: {issue}")
        logger.warning("   Fix: Start Ollama and run: ollama pull llama3.2")
    elif not model_ok:
        logger.warning(f"⚠️  Model '{settings.ollama_model}' is NOT available!")
        logger.warning(f"   Fix: ollama pull {settings.ollama_model}")
    else:
        logger.info(f"✅ Ollama ready | Model: {settings.ollama_model}")

    try:
        vs = get_vector_store()
        count = vs.get_total_chunk_count()
        logger.info(f"✅ ChromaDB ready | {count} chunks indexed")
    except Exception as e:
        logger.error(f"⚠️  ChromaDB initialization error: {e}")

    logger.info("🚀 Agentic RAG backend is ready!")
    logger.info("   Frontend: Open frontend/index.html in your browser")
    logger.info("   API Docs: http://localhost:8000/docs")

    yield

    logger.info("Agentic RAG backend shutting down...")

app = FastAPI(
    title="Agentic RAG AI Assistant",
    description=(
        "A Local Agentic RAG system with autonomous decision-making, "
        "tool calling, query rewriting, and retrieval validation. "
        "Powered by Ollama + LangGraph + ChromaDB."
    ),
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """
    Check the health of all system components.
    Returns status of Ollama, the model, and ChromaDB.
    """
    ollama_available, model_available, issues = await check_ollama()
    setup_instructions = []

    if not ollama_available:
        setup_instructions.append("Start Ollama: ollama serve")
    if not model_available:
        setup_instructions.append(f"Pull model: ollama pull {settings.ollama_model}")

    try:
        vs = get_vector_store()
        doc_count = vs.get_total_chunk_count()
        chroma_available = True
    except Exception as e:
        chroma_available = False
        doc_count = 0
        issues.append(f"ChromaDB error: {str(e)[:100]}")

    if ollama_available and model_available and chroma_available:
        status = "healthy"
    elif chroma_available:
        status = "degraded"
    else:
        status = "unhealthy"

    return HealthResponse(
        status=status,
        ollama_available=ollama_available,
        ollama_model=settings.ollama_model,
        model_available=model_available,
        chroma_available=chroma_available,
        document_count=doc_count,
        issues=issues,
        setup_instructions=setup_instructions
    )

@app.post("/upload", response_model=UploadResponse, tags=["Documents"])
async def upload_document(file: UploadFile = File(...)):
    """
    Upload and index a document (PDF, TXT, or DOCX).
    
    The file will be:
    1. Validated for type and size
    2. Saved to the data directory
    3. Loaded and text extracted
    4. Chunked
    5. Embedded and indexed in ChromaDB
    """
    logger.info(f"Upload request: {file.filename} ({file.size} bytes)")

    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    try:
        file_content = await file.read()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not read file: {e}")

    result = await ingest_document(file_content, file.filename)

    if result["success"]:
        doc_info = result.get("document", {})
        return UploadResponse(
            success=True,
            message=result.get("message", "Document uploaded successfully"),
            document=DocumentInfo(**doc_info) if doc_info else None
        )
    else:
        return UploadResponse(
            success=False,
            message="Upload failed",
            error=result.get("error", "Unknown error")
        )

@app.get("/documents", response_model=DocumentListResponse, tags=["Documents"])
async def list_documents():
    """List all uploaded and indexed documents."""
    docs = get_all_documents()
    doc_infos = [DocumentInfo(**d) for d in docs]

    return DocumentListResponse(
        documents=doc_infos,
        total_count=len(doc_infos)
    )

@app.delete("/documents/{document_name}", response_model=DeleteResponse, tags=["Documents"])
async def delete_document_endpoint(document_name: str):
    """Delete a document and remove its vectors from ChromaDB."""
    logger.info(f"Delete request: {document_name}")

    result = await delete_document(document_name)

    return DeleteResponse(
        success=result["success"],
        message=result["message"]
    )

@app.post("/chat", response_model=ChatResponse, tags=["Chat"])
async def chat(request: ChatRequest):
    """
    Send a message to the Agentic RAG AI assistant.
    
    The agent will:
    1. Analyze your question
    2. Decide whether to search documents or answer directly
    3. If searching: retrieve, evaluate, possibly rewrite and retry
    4. Generate a grounded, cited answer
    """
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    response = await process_chat(
        question=request.question,
        conversation_id=request.conversation_id,
        history=request.history
    )

    return response

@app.post("/search", response_model=SearchResponse, tags=["Search"])
async def search_documents_endpoint(request: SearchRequest):
    """
    Directly search the document vector store (bypasses the agent).
    Useful for testing retrieval quality.
    """
    top_k = request.top_k or settings.top_k

    try:
        chunks = retrieve(
            query=request.query,
            top_k=top_k,
            filter_source=request.document_name
        )

        results = [
            SourceCitation(
                document=c.source,
                page=c.page,
                snippet=c.content[:300] + ("..." if len(c.content) > 300 else ""),
                score=round(c.score, 4)
            )
            for c in chunks
        ]

        return SearchResponse(
            results=results,
            total_found=len(results),
            query_used=request.query
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {e}")

@app.post("/reindex/{document_name}", tags=["Documents"])
async def reindex_document(document_name: str):
    """Re-index an existing document (useful if embeddings changed)."""
    file_path = settings.data_dir / document_name

    if not file_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Document '{document_name}' not found on disk"
        )

    try:
        file_content = file_path.read_bytes()
        result = await ingest_document(file_content, document_name)
        
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Reindex failed: {e}")

@app.get("/document/{document_name}", tags=["Documents"])
async def get_document_info(document_name: str):
    """Get detailed information about a specific document."""
    vs = get_vector_store()
    
    is_indexed = vs.is_document_indexed(document_name)
    chunk_count = vs.get_chunk_count_for_document(document_name)
    
    file_path = settings.data_dir / document_name
    file_exists = file_path.exists()
    
    return {
        "name": document_name,
        "indexed": is_indexed,
        "chunk_count": chunk_count,
        "file_exists": file_exists,
        "file_size": file_path.stat().st_size if file_exists else None
    }

frontend_dir = Path(__file__).resolve().parent.parent / "frontend"
if frontend_dir.exists():
    app.mount("/app", StaticFiles(directory=str(frontend_dir), html=True), name="frontend")

@app.get("/", tags=["System"])
async def root():
    """API root — returns basic info."""
    return {
        "name": "Agentic RAG AI Assistant",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs",
        "health": "/health",
        "app": "/app"
    }
