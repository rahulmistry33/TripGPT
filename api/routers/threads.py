from typing import List
from fastapi import APIRouter, HTTPException, status

from api.models import (
    ThreadSummary,
    ThreadDetail,
    CreateThreadRequest,
    MessageSchema,
)
from api.dependencies import SessionManagerDep, TripGraphDep

router = APIRouter(prefix="/api/v1/users/{user_id}/threads", tags=["Threads"])


def _serialize_message(msg) -> MessageSchema:
    """Helper to convert LangChain BaseMessage object to MessageSchema dict."""
    content = getattr(msg, "content", str(msg))
    
    # Determine role (human -> user, ai -> assistant)
    msg_type = getattr(msg, "type", "ai")
    role = "user" if msg_type == "human" else "assistant"
    
    return MessageSchema(role=role, content=content)


@router.get("", response_model=List[ThreadSummary])
async def list_threads(
    user_id: str,
    session_mgr: SessionManagerDep,
):
    """
    List all trip planning threads belonging to a specific user.
    Ideal for rendering sidebar conversation histories in frontend UIs.
    """
    threads_data = session_mgr.list_user_threads(user_id)
    return [
        ThreadSummary(
            thread_id=t["thread_id"],
            user_id=t["user_id"],
            title=t["title"],
            created_at=t["created_at"],
            updated_at=t["updated_at"],
            is_complete=t.get("is_complete", False),
            destination=t.get("destination"),
        )
        for t in threads_data
    ]


@router.post("", response_model=ThreadSummary, status_code=status.HTTP_201_CREATED)
async def create_thread(
    user_id: str,
    body: CreateThreadRequest,
    session_mgr: SessionManagerDep,
):
    """
    Create a new trip planning thread for a user.
    Returns the newly created thread summary including thread_id.
    """
    thread_id = session_mgr.create_thread(user_id=user_id, title=body.title)
    meta = session_mgr.get_thread(user_id=user_id, thread_id=thread_id)
    
    return ThreadSummary(
        thread_id=meta["thread_id"],
        user_id=meta["user_id"],
        title=meta["title"],
        created_at=meta["created_at"],
        updated_at=meta["updated_at"],
        is_complete=meta["is_complete"],
        destination=meta.get("destination"),
    )


@router.get("/{thread_id}", response_model=ThreadDetail)
async def get_thread_detail(
    user_id: str,
    thread_id: str,
    session_mgr: SessionManagerDep,
    trip_graph: TripGraphDep,
):
    """
    Get full thread details including conversation message history and current extracted trip details.
    """
    meta = session_mgr.get_thread(user_id=user_id, thread_id=thread_id)
    if not meta:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Thread '{thread_id}' not found for user '{user_id}'",
        )

    # Scoped thread key in LangGraph memory saver
    composite_thread_id = f"{user_id}::{thread_id}"
    config = {"configurable": {"thread_id": composite_thread_id}}
    
    # Retrieve current state snapshot from LangGraph checkpointer memory
    state_snapshot = trip_graph.graph.get_state(config)
    
    messages = []
    trip_details = meta.get("trip_details", {})
    is_complete = meta.get("is_complete", False)
    missing_fields = meta.get("missing_fields", [])

    if state_snapshot and state_snapshot.values:
        raw_msgs = state_snapshot.values.get("messages", [])
        messages = [_serialize_message(m) for m in raw_msgs]
        if "trip_details" in state_snapshot.values:
            trip_details = state_snapshot.values["trip_details"]
            is_complete = trip_details.get("is_complete", False)
            missing_fields = trip_details.get("missing_fields", [])

    return ThreadDetail(
        thread_id=meta["thread_id"],
        user_id=meta["user_id"],
        title=meta["title"],
        trip_details=trip_details,
        is_complete=is_complete,
        missing_fields=missing_fields,
        messages=messages,
        created_at=meta["created_at"],
        updated_at=meta["updated_at"],
    )


@router.delete("/{thread_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_thread(
    user_id: str,
    thread_id: str,
    session_mgr: SessionManagerDep,
):
    """
    Delete a trip planning thread for a user.
    """
    deleted = session_mgr.delete_thread(user_id=user_id, thread_id=thread_id)
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Thread '{thread_id}' not found for user '{user_id}'",
        )
    return None
