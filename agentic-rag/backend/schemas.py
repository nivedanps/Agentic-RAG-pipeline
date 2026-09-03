from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, List, TypedDict, Any
from datetime import datetime

class DocumentInfo(BaseModel):
    """Information about an uploaded document."""
    name: str = Field(description="Document filename")
    size_bytes: int = Field(description="File size in bytes")
    uploaded_at: str = Field(description="ISO timestamp of upload")
    status: str = Field(description="Processing status: pending, processing, ready, error")
    chunk_count: int = Field(default=0, description="Number of chunks indexed in ChromaDB")
    error_message: Optional[str] = Field(default=None, description="Error details if processing failed")

class DocumentListResponse(BaseModel):
    """Response for GET /documents."""
    documents: list[DocumentInfo]
    total_count: int

class UploadResponse(BaseModel):
    """Response for POST /upload."""
    success: bool
    message: str
    document: Optional[DocumentInfo] = None
    error: Optional[str] = None

class DeleteResponse(BaseModel):
    """Response for DELETE /documents/{document_name}."""
    success: bool
    message: str

class SourceCitation(BaseModel):
    """A single source citation pointing to a document chunk."""
    document: str = Field(description="Source document filename")
    page: Optional[int] = Field(default=None, description="Page number if applicable")
    snippet: str = Field(description="Relevant text snippet from the source")
    score: Optional[float] = Field(default=None, description="Relevance score (0-1)")

class ChatMessage(BaseModel):
    """A single message in the conversation."""
    role: str = Field(description="'user' or 'assistant'")
    content: str = Field(description="Message content")
    timestamp: Optional[str] = Field(default=None, description="ISO timestamp")

class ChatRequest(BaseModel):
    """Request body for POST /chat."""
    question: str = Field(description="User's question", min_length=1, max_length=2000)
    conversation_id: Optional[str] = Field(
        default=None,
        description="Conversation ID to maintain history. If None, starts a new conversation."
    )
    history: Optional[list[ChatMessage]] = Field(
        default=None,
        description="Conversation history sent from the frontend"
    )

class AgentTrace(BaseModel):
    """High-level agent activity trace — safe to expose to the frontend."""
    step: str = Field(description="Step name/identifier")
    message: str = Field(description="Human-readable description of what the agent is doing")
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())

class ChatResponse(BaseModel):
    """Response for POST /chat."""
    answer: str = Field(description="Final answer from the AI agent")
    sources: list[SourceCitation] = Field(
        default_factory=list,
        description="Source citations (empty for direct answers)"
    )
    agent_action: str = Field(
        description="What the agent decided: 'direct_answer' or 'document_search'"
    )
    tools_used: list[str] = Field(
        default_factory=list,
        description="List of tool names the agent called"
    )
    retrieval_attempts: int = Field(
        default=0,
        description="Number of retrieval attempts made"
    )
    agent_trace: list[AgentTrace] = Field(
        default_factory=list,
        description="High-level activity trace for the UI agent panel"
    )
    conversation_id: Optional[str] = Field(
        default=None,
        description="Conversation ID for follow-up messages"
    )

class SearchRequest(BaseModel):
    """Request body for POST /search."""
    query: str = Field(description="Search query", min_length=1)
    document_name: Optional[str] = Field(
        default=None,
        description="If provided, search only within this document"
    )
    top_k: Optional[int] = Field(
        default=None,
        description="Number of results to return (uses config default if None)"
    )

class SearchResponse(BaseModel):
    """Response for POST /search."""
    results: list[SourceCitation]
    total_found: int
    query_used: str

class HealthResponse(BaseModel):
    """Response for GET /health."""
    model_config = ConfigDict(protected_namespaces=())

    status: str = Field(description="'healthy', 'degraded', or 'unhealthy'")
    ollama_available: bool
    ollama_model: str
    model_available: bool
    chroma_available: bool
    document_count: int
    issues: list[str] = Field(default_factory=list)
    setup_instructions: list[str] = Field(default_factory=list)

class AgentState(TypedDict):
    """The complete state of the agent during one question-answering cycle."""
    user_question: str
    conversation_history: list
    requires_retrieval: bool
    decision_reasoning: str
    tool_called: str
    tools_used: List[str]
    search_query: str
    retrieved_chunks: list
    retrieval_attempts: int
    context_sufficient: bool
    final_answer: str
    sources: list
    agent_action: str
    agent_trace: List[dict]
    error: Optional[str]

def create_initial_state(
    user_question: str,
    conversation_history: list = None
) -> AgentState:
    return AgentState(
        user_question=user_question,
        conversation_history=conversation_history or [],
        requires_retrieval=False,
        decision_reasoning="",
        tool_called="",
        tools_used=[],
        search_query=user_question,
        retrieved_chunks=[],
        retrieval_attempts=0,
        context_sufficient=False,
        final_answer="",
        sources=[],
        agent_action="direct_answer",
        agent_trace=[],
        error=None
    )

def add_trace(state: AgentState, step: str, message: str) -> None:
    trace_entry = {
        "step": step,
        "message": message,
        "timestamp": datetime.now().isoformat()
    }
    state["agent_trace"].append(trace_entry)
