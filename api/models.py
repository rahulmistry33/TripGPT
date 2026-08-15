from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime


class ChatRequest(BaseModel):
    """Payload sent by the frontend UI when a user sends a chat message."""
    user_id: str = Field(
        ...,
        description="Unique identifier for the user (e.g. 'user_123').",
        examples=["user_123"]
    )
    thread_id: Optional[str] = Field(
        default=None,
        description="ID of the trip conversation thread. If omitted, a new thread is created automatically.",
        examples=["trip_abc123"]
    )
    message: str = Field(
        ...,
        description="The message text from the user.",
        examples=["I want to plan a 5-day trip to Tokyo from Mumbai in October with a $3000 budget for 2 people."]
    )


class ChatResponse(BaseModel):
    """Payload returned to the frontend UI after an agent execution turn."""
    user_id: str = Field(..., description="User ID for the session.")
    thread_id: str = Field(..., description="Thread ID for the session.")
    reply: str = Field(..., description="The assistant's text response.")
    trip_details: Dict[str, Any] = Field(..., description="Current state of extracted trip details.")
    is_complete: bool = Field(..., description="Whether all required trip fields are extracted and fanned out.")
    missing_fields: List[str] = Field(..., description="List of trip fields still required.")
    updated_at: str = Field(..., description="ISO timestamp of this response turn.")
    awaiting_approval: bool = Field(default=False, description="True if graph is currently paused for human approval (HITL).")
    guardrail_status: Optional[Dict[str, Any]] = Field(default=None, description="Guardrail validation results.")
    summary_for_approval: Optional[str] = Field(default=None, description="Trip proposal summary awaiting human approval.")


class ResumeRequest(BaseModel):
    """Payload sent by the user/UI to approve or request revisions during HITL graph interrupt."""
    user_id: str = Field(..., description="User ID for the session.", examples=["user_123"])
    thread_id: str = Field(..., description="Thread ID of the paused trip conversation.", examples=["trip_abc123"])
    approved: bool = Field(default=True, description="Set to True to approve options, or False to reject with feedback.")
    feedback: Optional[str] = Field(default=None, description="Optional revision instructions if rejected.", examples=["Find cheaper hotels under $200/night."])



class CreateThreadRequest(BaseModel):
    """Payload for manually creating a new trip conversation thread."""
    title: Optional[str] = Field(
        default=None,
        description="Optional title for the trip thread (e.g. 'Tokyo Vacation 2026')."
    )


class ThreadSummary(BaseModel):
    """Summary of a trip thread for list views in the UI sidebar."""
    thread_id: str
    user_id: str
    title: str
    created_at: str
    updated_at: str
    is_complete: bool
    destination: Optional[str] = None


class MessageSchema(BaseModel):
    """Single chat message schema for conversation history."""
    role: str = Field(..., description="'user' or 'assistant'")
    content: str = Field(..., description="Message text content")


class ThreadDetail(BaseModel):
    """Full detail of a thread including state and message history."""
    thread_id: str
    user_id: str
    title: str
    trip_details: Dict[str, Any]
    is_complete: bool
    missing_fields: List[str]
    messages: List[MessageSchema]
    created_at: str
    updated_at: str
