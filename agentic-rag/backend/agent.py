import logging
import uuid
from datetime import datetime
from typing import Literal, Optional, List

# pyrefly: ignore [missing-import]
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage
# pyrefly: ignore [missing-import]
from langgraph.graph import StateGraph, END

from backend.config import settings
from backend.schemas import (
    AgentState, create_initial_state, add_trace,
    ChatResponse, SourceCitation, AgentTrace
)
from backend.rag import (
    get_vector_store, retrieve, format_chunks_for_context, RetrievedChunk
)

logger = logging.getLogger("agentic_rag.agent")

DECISION_PROMPT = """You are an AI assistant analyzing a user's input.

Your task: Decide whether the user's question REQUIRES searching uploaded documents, or can be answered DIRECTLY.

RULES FOR "DIRECT":
- Greetings or casual conversation (e.g., "Hello", "Hi", "How are you", "Good morning") -> DIRECT
- General knowledge questions NOT related to specific files (e.g., "What is the capital of France?", "Who wrote Hamlet?") -> DIRECT
- System capability questions (e.g., "What can you do?") -> DIRECT

RULES FOR "RETRIEVE":
- Questions specifically asking about company metrics, financial reports, uploaded documents, files, PDFs, data, or facts -> RETRIEVE
- Questions asking to summarize, compare, or extract details from documents -> RETRIEVE

USER QUESTION: {question}

Available documents: {document_list}

Respond ONLY in this exact format:
[DECISION]
Reasoning sentence.

Where [DECISION] is either DIRECT or RETRIEVE.
"""

DIRECT_ANSWER_PROMPT = """You are a helpful AI assistant.

Answer the user's question directly using your knowledge.
Be concise and accurate.

{conversation_context}

USER QUESTION: {question}

Answer:"""

RELEVANCE_EVALUATION_PROMPT = """You are evaluating whether retrieved document context is sufficient to answer a question.

USER QUESTION: {question}

RETRIEVED CONTEXT:
{context}

EVALUATION TASK:
1. Is the retrieved context relevant to the question? (Does it contain information about this topic?)
2. Is the context sufficient to provide a complete, accurate answer?
3. Would searching with a different query find better information?

Respond with ONLY:
SUFFICIENT
or
INSUFFICIENT

Then on the next line, provide a better search query if INSUFFICIENT (or "N/A" if SUFFICIENT).
Keep the alternative query concise and specific (max 10 words).
"""

QUERY_REWRITE_PROMPT = """You are a search query optimizer.

The original question did not return sufficient information from the document database.
Create a better, more specific search query to find the relevant information.

ORIGINAL QUESTION: {original_question}
PREVIOUS SEARCH QUERY: {previous_query}
REASON FOR REWRITING: {reason}

Rules for the new query:
- Make it more specific and focused
- Use key terms likely to appear in the document
- Avoid question words (how, what, why) — use keyword phrases instead
- Keep it under 10 words

NEW SEARCH QUERY (respond with ONLY the query, nothing else):"""

GROUNDED_ANSWER_PROMPT = """You are a precise AI assistant that answers questions based on document evidence.

IMPORTANT RULES:
1. Answer ONLY based on the provided context — do not add information not in the context
2. If the context does not contain enough information, clearly say so
3. Always cite your sources (document name and page if available)
4. Be concise and direct
5. Do not speculate or hallucinate

{conversation_context}

USER QUESTION: {question}

RETRIEVED DOCUMENT CONTEXT:
{context}

Instructions:
- If the context answers the question: Provide a clear, accurate answer
- If the context partially answers: Answer what you can, note what is missing
- If the context is unrelated: Say "I couldn't find sufficient information in the uploaded documents to answer this accurately."

ANSWER:"""


def format_conversation_context(history: list, max_turns: int = 5) -> str:
    if not history:
        return ""

    recent = history[-max_turns * 2:]
    if not recent:
        return ""

    lines = ["CONVERSATION HISTORY:"]
    for msg in recent:
        role = msg.get("role", "unknown").upper()
        content = msg.get("content", "")
        lines.append(f"{role}: {content}")

    return "\n".join(lines) + "\n"


def search_documents(query: str, top_k: int = None) -> dict:
    logger.info(f"TOOL CALLED: search_documents(query='{query[:60]}')")
    top_k = top_k or settings.top_k

    try:
        chunks = retrieve(query=query, top_k=top_k)
        if not chunks:
            logger.info("search_documents: No results found")
            return {
                "context": "No relevant information was found in the uploaded documents for this query.",
                "chunks": [],
                "found": False,
                "chunk_count": 0
            }

        context = format_chunks_for_context(chunks)
        serialized_chunks = [
            {
                "content": c.content,
                "source": c.source,
                "page": c.page,
                "score": round(c.score, 4)
            }
            for c in chunks
        ]

        logger.info(f"search_documents: Found {len(chunks)} relevant chunks")
        return {
            "context": context,
            "chunks": serialized_chunks,
            "found": True,
            "chunk_count": len(chunks)
        }
    except Exception as e:
        logger.error(f"search_documents tool error: {e}")
        return {
            "context": f"Document search encountered an error: {str(e)}",
            "chunks": [],
            "found": False,
            "chunk_count": 0
        }


def list_documents() -> dict:
    logger.info("TOOL CALLED: list_documents()")
    try:
        vector_store = get_vector_store()
        indexed = vector_store.list_indexed_documents()

        data_dir = settings.data_dir
        uploaded_files = list(data_dir.glob("*"))
        uploaded_names = {f.name for f in uploaded_files if f.is_file() and not f.name.startswith(".")}

        documents = []
        for item in indexed:
            source = item["source"]
            doc_info = {
                "name": source,
                "chunk_count": item["chunk_count"],
                "file_exists": source in uploaded_names,
                "indexed": True
            }
            documents.append(doc_info)

        if not documents:
            message = "No documents have been uploaded and indexed yet."
        else:
            doc_names = [d["name"] for d in documents]
            message = f"Available documents ({len(documents)}): {', '.join(doc_names)}"

        logger.info(f"list_documents: Found {len(documents)} documents")
        return {
            "documents": documents,
            "count": len(documents),
            "message": message
        }
    except Exception as e:
        logger.error(f"list_documents tool error: {e}")
        return {
            "documents": [],
            "count": 0,
            "message": f"Could not retrieve document list: {str(e)}"
        }


def search_specific_document(document_name: str, query: str, top_k: int = None) -> dict:
    logger.info(
        f"TOOL CALLED: search_specific_document(document='{document_name}', query='{query[:60]}')"
    )
    top_k = top_k or settings.top_k

    try:
        chunks = retrieve(
            query=query,
            top_k=top_k,
            filter_source=document_name
        )

        if not chunks:
            return {
                "context": f"No relevant information found in '{document_name}' for this query.",
                "chunks": [],
                "found": False,
                "chunk_count": 0,
                "document_name": document_name
            }

        context = format_chunks_for_context(chunks)
        serialized_chunks = [
            {
                "content": c.content,
                "source": c.source,
                "page": c.page,
                "score": round(c.score, 4)
            }
            for c in chunks
        ]

        logger.info(f"search_specific_document: Found {len(chunks)} chunks in '{document_name}'")
        return {
            "context": context,
            "chunks": serialized_chunks,
            "found": True,
            "chunk_count": len(chunks),
            "document_name": document_name
        }
    except Exception as e:
        logger.error(f"search_specific_document tool error: {e}")
        return {
            "context": f"Error searching '{document_name}': {str(e)}",
            "chunks": [],
            "found": False,
            "chunk_count": 0,
            "document_name": document_name
        }


def document_summary(document_name: str) -> dict:
    logger.info(f"TOOL CALLED: document_summary(document='{document_name}')")

    try:
        vector_store = get_vector_store()
        all_chunks_result = vector_store.get_document_chunks(document_name)

        if not all_chunks_result["documents"]:
            return {
                "context": f"No content found for document '{document_name}'. It may not be indexed.",
                "found": False,
                "chunk_count": 0,
                "document_name": document_name
            }

        texts = all_chunks_result["documents"]
        metadatas = all_chunks_result["metadatas"]
        total_chunks = len(texts)

        sample_size = min(8, total_chunks)
        if total_chunks <= sample_size:
            indices = list(range(total_chunks))
        else:
            step = total_chunks // sample_size
            indices = [i * step for i in range(sample_size)]
            indices[-1] = total_chunks - 1

        chunks = []
        for idx in indices:
            meta = metadatas[idx]
            chunk = RetrievedChunk(
                content=texts[idx],
                source=meta.get("source", document_name),
                page=meta.get("page", None),
                score=1.0,
                chunk_index=meta.get("chunk_index", idx)
            )
            chunks.append(chunk)

        context = format_chunks_for_context(chunks)
        logger.info(
            f"document_summary: Selected {len(chunks)}/{total_chunks} "
            f"chunks for '{document_name}'"
        )
        return {
            "context": context,
            "found": True,
            "chunk_count": len(chunks),
            "total_chunks": total_chunks,
            "document_name": document_name
        }
    except Exception as e:
        logger.error(f"document_summary tool error: {e}")
        return {
            "context": f"Error summarizing '{document_name}': {str(e)}",
            "found": False,
            "chunk_count": 0,
            "document_name": document_name
        }


def get_llm() -> ChatOllama:
    return ChatOllama(
        base_url=settings.ollama_base_url,
        model=settings.ollama_model,
        temperature=0.1,
        timeout=120
    )


def _call_llm(prompt: str) -> str:
    try:
        llm = get_llm()
        response = llm.invoke([HumanMessage(content=prompt)])
        return response.content.strip()
    except Exception as e:
        logger.error(f"LLM call failed: {e}")
        raise RuntimeError(f"Ollama LLM error: {e}")


def analyze_question_node(state: AgentState) -> AgentState:
    logger.info(f"NODE: analyze_question | question='{state['user_question'][:60]}'")
    add_trace(state, "analyzing", "🧠 Analyzing your question...")

    clean_q = state["user_question"].strip().lower().rstrip(".!?")
    greetings = ["hello", "hi", "hey", "good morning", "good afternoon", "good evening", "who are you", "what can you do", "help"]
    if clean_q in greetings:
        state["requires_retrieval"] = False
        state["decision_reasoning"] = "User input is a greeting or general greeting"
        state["agent_action"] = "direct_answer"
        add_trace(state, "decision", "💡 Decision: Answering from general knowledge")
        logger.info("Decision: DIRECT | Reason: Greeting detected")
        return state

    try:
        docs_result = list_documents()
        doc_list = docs_result.get("message", "No documents available")

        prompt = DECISION_PROMPT.format(
            question=state["user_question"],
            document_list=doc_list
        )

        response = _call_llm(prompt)
        first_line = response.strip().split("\n")[0].upper()

        if "DIRECT" in first_line:
            requires_retrieval = False
        elif "RETRIEVE" in first_line:
            requires_retrieval = True
        elif "DIRECT" in response.upper() and ("RETRIEVE" not in response.upper() or response.upper().find("DIRECT") < response.upper().find("RETRIEVE")):
            requires_retrieval = False
        else:
            requires_retrieval = True

        lines = response.strip().split("\n", 1)
        reasoning = lines[1].strip() if len(lines) > 1 else response.strip()

        state["requires_retrieval"] = requires_retrieval
        state["decision_reasoning"] = reasoning

        if requires_retrieval:
            state["agent_action"] = "document_search"
            add_trace(state, "decision", "🔍 Decision: Searching uploaded documents")
            logger.info(f"Decision: RETRIEVE | Reason: {reasoning}")
        else:
            state["agent_action"] = "direct_answer"
            add_trace(state, "decision", "💡 Decision: Answering from general knowledge")
            logger.info(f"Decision: DIRECT | Reason: {reasoning}")

    except Exception as e:
        logger.error(f"analyze_question_node error: {e}")
        state["requires_retrieval"] = False
        state["agent_action"] = "direct_answer"
        state["error"] = str(e)
        add_trace(state, "error", "⚠️ Analysis error, attempting direct response")

    return state


def generate_direct_answer_node(state: AgentState) -> AgentState:
    logger.info("NODE: generate_direct_answer")
    add_trace(state, "generating", "✍️ Generating direct answer...")

    try:
        conv_context = format_conversation_context(state["conversation_history"])
        prompt = DIRECT_ANSWER_PROMPT.format(
            question=state["user_question"],
            conversation_context=conv_context
        )

        answer = _call_llm(prompt)
        state["final_answer"] = answer
        state["sources"] = []
        add_trace(state, "complete", "✅ Answer ready")

    except Exception as e:
        logger.error(f"generate_direct_answer_node error: {e}")
        state["final_answer"] = (
            "I apologize, but I encountered an error generating a response. "
            "Please check that Ollama is running and the model is available."
        )
        state["error"] = str(e)

    return state


def retrieve_documents_node(state: AgentState) -> AgentState:
    logger.info(
        f"NODE: retrieve_documents | "
        f"attempt={state['retrieval_attempts'] + 1} | "
        f"query='{state['search_query'][:60]}'"
    )

    attempt_num = state["retrieval_attempts"] + 1
    add_trace(state, "retrieval", f"🔍 Searching documents (attempt {attempt_num})...")

    try:
        result = search_documents(state["search_query"])
        state["retrieved_chunks"] = result.get("chunks", [])
        state["retrieval_attempts"] += 1

        if "search_documents" not in state["tools_used"]:
            state["tools_used"].append("search_documents")
        state["tool_called"] = "search_documents"

        chunk_count = result.get("chunk_count", 0)
        if chunk_count > 0:
            add_trace(state, "retrieved", f"📄 Retrieved {chunk_count} relevant chunks")
        else:
            add_trace(state, "retrieved", "📭 No matching content found")

        logger.info(f"Retrieved {chunk_count} chunks")

    except Exception as e:
        logger.error(f"retrieve_documents_node error: {e}")
        state["retrieved_chunks"] = []
        state["retrieval_attempts"] += 1
        state["error"] = str(e)
        add_trace(state, "error", f"⚠️ Retrieval error: {str(e)[:50]}")

    return state


def evaluate_context_node(state: AgentState) -> AgentState:
    logger.info("NODE: evaluate_context")
    add_trace(state, "evaluating", "🔎 Evaluating retrieved context quality...")

    chunks = state.get("retrieved_chunks", [])
    if not chunks:
        state["context_sufficient"] = False
        logger.info("Context evaluation: INSUFFICIENT (no chunks found)")
        return state

    context_text = "\n\n".join([
        f"[{c.get('source', 'unknown')}"
        f"{', Page ' + str(c['page']) if c.get('page') else ''}]\n"
        f"{c['content']}"
        for c in chunks
    ])

    try:
        prompt = RELEVANCE_EVALUATION_PROMPT.format(
            question=state["user_question"],
            context=context_text[:3000]
        )

        response = _call_llm(prompt)
        lines = response.strip().split("\n", 1)
        verdict = lines[0].strip().upper()

        is_sufficient = "SUFFICIENT" in verdict and "INSUFFICIENT" not in verdict
        state["context_sufficient"] = is_sufficient

        if is_sufficient:
            add_trace(state, "evaluation", " Context is sufficient")
            logger.info("Context evaluation: SUFFICIENT")
        else:
            add_trace(state, "evaluation", "🔄 Context insufficient, will refine search")
            logger.info("Context evaluation: INSUFFICIENT")

    except Exception as e:
        logger.error(f"evaluate_context_node error: {e}")
        state["context_sufficient"] = True
        add_trace(state, "evaluation", "✅ Proceeding with available context")

    return state


def rewrite_query_node(state: AgentState) -> AgentState:
    logger.info(f"NODE: rewrite_query | previous='{state['search_query'][:60]}'")
    add_trace(state, "rewriting", "🔄 Rewriting search query for better results...")

    try:
        prompt = QUERY_REWRITE_PROMPT.format(
            original_question=state["user_question"],
            previous_query=state["search_query"],
            reason="Previous search did not find sufficient relevant information"
        )

        new_query = _call_llm(prompt)
        new_query = new_query.strip().strip('"').strip("'")
        logger.info(f"Query rewritten: '{state['search_query']}' → '{new_query}'")
        state["search_query"] = new_query
        add_trace(state, "rewritten", f"✏️ New query: '{new_query[:60]}'")

    except Exception as e:
        logger.error(f"rewrite_query_node error: {e}")
        state["search_query"] = state["user_question"] + " details information"
        add_trace(state, "rewritten", "✏️ Query adjusted for retry")

    return state


def generate_grounded_answer_node(state: AgentState) -> AgentState:
    logger.info("NODE: generate_grounded_answer")
    add_trace(state, "generating", "✍️ Generating grounded answer from documents...")

    chunks = state.get("retrieved_chunks", [])
    if not chunks:
        state["final_answer"] = (
            "I couldn't find sufficient information in the uploaded documents "
            "to answer this question accurately. Please ensure relevant documents "
            "have been uploaded, or try rephrasing your question."
        )
        state["sources"] = []
        add_trace(state, "complete", "✅ Response generated (no relevant content found)")
        return state

    context_parts = []
    sources = []
    seen_sources = set()

    for chunk in chunks:
        source = chunk.get("source", "unknown")
        page = chunk.get("page")
        content = chunk.get("content", "")
        score = chunk.get("score", 0)

        source_label = f"[{source}" + (f", Page {page}" if page else "") + "]"
        context_parts.append(f"{source_label}\n{content}")

        source_key = f"{source}_{page}"
        if source_key not in seen_sources:
            seen_sources.add(source_key)
            sources.append({
                "document": source,
                "page": page,
                "snippet": content[:200] + ("..." if len(content) > 200 else ""),
                "score": score
            })

    context_text = "\n\n".join(context_parts)
    conv_context = format_conversation_context(state["conversation_history"])

    try:
        prompt = GROUNDED_ANSWER_PROMPT.format(
            question=state["user_question"],
            context=context_text,
            conversation_context=conv_context
        )

        answer = _call_llm(prompt)
        state["final_answer"] = answer
        state["sources"] = sources
        add_trace(state, "complete", f"✅ Answer generated with {len(sources)} source(s)")

    except Exception as e:
        logger.error(f"generate_grounded_answer_node error: {e}")
        state["final_answer"] = (
            "I encountered an error while generating the answer. "
            "Please check that Ollama is running properly."
        )
        state["error"] = str(e)
        add_trace(state, "error", "⚠️ Error generating answer")

    return state


def should_retrieve(state: AgentState) -> Literal["retrieve", "direct"]:
    if state.get("requires_retrieval", False):
        return "retrieve"
    return "direct"


def is_context_sufficient(state: AgentState) -> Literal["answer", "rewrite", "max_attempts"]:
    max_attempts = settings.max_retrieval_attempts
    attempts = state.get("retrieval_attempts", 0)

    if state.get("context_sufficient", False):
        return "answer"

    if attempts >= max_attempts:
        logger.warning(
            f"Max retrieval attempts ({max_attempts}) reached. "
            f"Proceeding with available context."
        )
        return "max_attempts"

    return "rewrite"


def build_agent_graph() -> StateGraph:
    graph = StateGraph(AgentState)

    graph.add_node("analyze_question", analyze_question_node)
    graph.add_node("retrieve_documents", retrieve_documents_node)
    graph.add_node("evaluate_context", evaluate_context_node)
    graph.add_node("rewrite_query", rewrite_query_node)
    graph.add_node("generate_direct_answer", generate_direct_answer_node)
    graph.add_node("generate_grounded_answer", generate_grounded_answer_node)

    graph.set_entry_point("analyze_question")

    graph.add_conditional_edges(
        "analyze_question",
        should_retrieve,
        {
            "retrieve": "retrieve_documents",
            "direct": "generate_direct_answer"
        }
    )

    graph.add_edge("generate_direct_answer", END)
    graph.add_edge("retrieve_documents", "evaluate_context")

    graph.add_conditional_edges(
        "evaluate_context",
        is_context_sufficient,
        {
            "answer": "generate_grounded_answer",
            "rewrite": "rewrite_query",
            "max_attempts": "generate_grounded_answer"
        }
    )

    graph.add_edge("rewrite_query", "retrieve_documents")
    graph.add_edge("generate_grounded_answer", END)

    compiled = graph.compile()
    logger.info("LangGraph agentic workflow compiled successfully")
    return compiled


_agent_graph = None


def get_agent_graph():
    global _agent_graph
    if _agent_graph is None:
        _agent_graph = build_agent_graph()
    return _agent_graph


def run_agent(
    question: str,
    conversation_history: list = None
) -> AgentState:
    logger.info(f"=== AGENT RUN START | question='{question[:80]}' ===")

    initial_state = create_initial_state(
        user_question=question,
        conversation_history=conversation_history or []
    )

    graph = get_agent_graph()

    try:
        final_state = graph.invoke(initial_state)
        logger.info(
            f"=== AGENT RUN COMPLETE | "
            f"action={final_state.get('agent_action')} | "
            f"attempts={final_state.get('retrieval_attempts', 0)} | "
            f"sources={len(final_state.get('sources', []))} ==="
        )
        return final_state

    except Exception as e:
        logger.error(f"Agent run failed: {e}")
        initial_state["final_answer"] = (
            "I encountered an unexpected error. "
            "Please check that Ollama is running and try again."
        )
        initial_state["error"] = str(e)
        initial_state["agent_action"] = "error"
        return initial_state


_conversations: dict = {}


def create_conversation() -> str:
    conv_id = str(uuid.uuid4())
    _conversations[conv_id] = []
    return conv_id


def get_conversation_history(conversation_id: str) -> list:
    return _conversations.get(conversation_id, [])


def add_to_history(conversation_id: str, role: str, content: str) -> None:
    if conversation_id not in _conversations:
        _conversations[conversation_id] = []

    _conversations[conversation_id].append({
        "role": role,
        "content": content,
        "timestamp": datetime.now().isoformat()
    })

    max_messages = settings.max_history_length * 2
    if len(_conversations[conversation_id]) > max_messages:
        _conversations[conversation_id] = _conversations[conversation_id][-max_messages:]


def clear_conversation(conversation_id: str) -> None:
    if conversation_id in _conversations:
        _conversations[conversation_id] = []


async def process_chat(
    question: str,
    conversation_id: Optional[str] = None,
    history: Optional[list] = None
) -> ChatResponse:
    logger.info(f"Processing chat: question='{question[:60]}', session={conversation_id}")

    if not conversation_id:
        conversation_id = create_conversation()
        logger.info(f"New conversation created: {conversation_id}")

    if history:
        conversation_history = [
            {
                "role": getattr(msg, "role", None) if not isinstance(msg, dict) else msg.get("role"),
                "content": getattr(msg, "content", None) if not isinstance(msg, dict) else msg.get("content")
            }
            for msg in history
        ]
    else:
        conversation_history = get_conversation_history(conversation_id)

    try:
        final_state = run_agent(
            question=question,
            conversation_history=conversation_history
        )
    except Exception as e:
        logger.error(f"Agent execution error: {e}")
        return ChatResponse(
            answer=(
                "I encountered an error processing your request. "
                "Please check that Ollama is running and try again. "
                f"Error: {str(e)}"
            ),
            sources=[],
            agent_action="error",
            tools_used=[],
            retrieval_attempts=0,
            agent_trace=[AgentTrace(
                step="error",
                message=f"⚠️ System error: {str(e)[:100]}"
            )],
            conversation_id=conversation_id
        )

    add_to_history(conversation_id, "user", question)
    add_to_history(conversation_id, "assistant", final_state.get("final_answer", ""))

    sources = [
        SourceCitation(
            document=s.get("document", "unknown"),
            page=s.get("page"),
            snippet=s.get("snippet", ""),
            score=s.get("score")
        )
        for s in final_state.get("sources", [])
    ]

    trace = [
        AgentTrace(
            step=t.get("step", ""),
            message=t.get("message", ""),
            timestamp=t.get("timestamp", datetime.now().isoformat())
        )
        for t in final_state.get("agent_trace", [])
    ]

    response = ChatResponse(
        answer=final_state.get("final_answer", "No answer generated"),
        sources=sources,
        agent_action=final_state.get("agent_action", "unknown"),
        tools_used=final_state.get("tools_used", []),
        retrieval_attempts=final_state.get("retrieval_attempts", 0),
        agent_trace=trace,
        conversation_id=conversation_id
    )

    logger.info(
        f"Chat response ready: action={response.agent_action}, "
        f"sources={len(sources)}, attempts={response.retrieval_attempts}"
    )
    return response
