#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -d "$SCRIPT_DIR/agentic-rag" ]; then
    cd "$SCRIPT_DIR/agentic-rag"
else
    cd "$SCRIPT_DIR"
fi
source .venv/bin/activate
uvicorn backend.main:app --reload --port 8000
