"""Session management endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from faqih.api.dependencies import get_conversation_memory
from faqih.memory.conversation import ConversationMemory

router = APIRouter()


@router.post("")
async def create_session(
    memory: ConversationMemory = Depends(get_conversation_memory),
):
    """Create a new conversation session."""
    session = await memory.create_session()
    return {
        "session_id": session.session_id,
        "created_at": session.created_at.isoformat(),
    }


@router.get("/{session_id}/history")
async def get_history(
    session_id: str,
    memory: ConversationMemory = Depends(get_conversation_memory),
):
    """Get conversation history for a session."""
    turns = await memory.get_recent_turns(session_id)
    return {
        "session_id": session_id,
        "turns": [t.model_dump(mode="json") for t in turns],
        "entity_history": await memory.get_entity_history(session_id),
    }
