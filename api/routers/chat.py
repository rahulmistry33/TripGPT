import asyncio
from datetime import datetime, timezone
from fastapi import APIRouter, HTTPException, status
from langgraph.errors import GraphInterrupt


from api.models import ChatRequest, ChatResponse, ResumeRequest
from api.dependencies import SessionManagerDep, TripGraphDep

router = APIRouter(prefix="/api/v1/chat", tags=["Chat"])


def _extract_response_data(
    result: dict,
    state_snapshot,
    user_id: str,
    thread_id: str,
    session_mgr,
) -> ChatResponse:
    """Helper to extract ChatResponse from graph result and state snapshot."""
    # Check if graph is currently halted at an interrupt point
    awaiting_approval = False
    summary_for_approval = None

    if state_snapshot and state_snapshot.tasks:
        for task in state_snapshot.tasks:
            if hasattr(task, "interrupts") and task.interrupts:
                for intr in task.interrupts:
                    intr_val = getattr(intr, "value", {})
                    if isinstance(intr_val, dict) and intr_val.get("action") == "human_approval_required":
                        awaiting_approval = True
                        summary_for_approval = intr_val.get("summary")
                        break

    # Extract messages and reply text
    messages = result.get("messages", [])
    reply_text = ""
    if messages:
        latest_msg = messages[-1]
        reply_text = getattr(latest_msg, "content", str(latest_msg))
    elif awaiting_approval and summary_for_approval:
        reply_text = summary_for_approval

    guardrail_status = result.get("guardrail_status")

    trip_details = result.get("trip_details", {})
    if isinstance(trip_details, dict):
        is_complete = trip_details.get("is_complete", False)
        missing_fields = trip_details.get("missing_fields", [])
    else:
        is_complete = getattr(trip_details, "is_complete", False)
        missing_fields = getattr(trip_details, "missing_fields", [])
        trip_details = trip_details.model_dump() if hasattr(trip_details, "model_dump") else dict(trip_details)

    # Update session metadata
    session_mgr.update_thread_state(
        user_id=user_id,
        thread_id=thread_id,
        trip_details=trip_details,
    )

    now_iso = datetime.now(timezone.utc).isoformat()

    return ChatResponse(
        user_id=user_id,
        thread_id=thread_id,
        reply=reply_text,
        trip_details=trip_details,
        is_complete=is_complete,
        missing_fields=missing_fields,
        updated_at=now_iso,
        awaiting_approval=awaiting_approval,
        guardrail_status=guardrail_status,
        summary_for_approval=summary_for_approval,
    )


@router.post("", response_model=ChatResponse)
async def chat_turn(
    req: ChatRequest,
    session_mgr: SessionManagerDep,
    trip_graph: TripGraphDep,
):
    """
    Send a message turn to the trip-gpt multi-agent system.
    
    - Runs Input/Output Guardrails and Multi-Agent Planning.
    - If HITL approval is required, graph halts and returns `awaiting_approval: true`.
    """
    user_id = req.user_id
    
    if not req.thread_id:
        thread_id = session_mgr.create_thread(user_id=user_id)
    else:
        thread_id = req.thread_id
        session_mgr.ensure_thread_exists(user_id=user_id, thread_id=thread_id)

    composite_thread_id = f"{user_id}::{thread_id}"

    try:
        result = await asyncio.to_thread(
            trip_graph.run_turn,
            user_input=req.message,
            thread_id=composite_thread_id,
        )
    except GraphInterrupt:
        # LangGraph raises GraphInterrupt when interrupt() is executed inside a node
        config = {"configurable": {"thread_id": composite_thread_id}}
        state_snapshot = trip_graph.graph.get_state(config)
        result = state_snapshot.values
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred while executing trip graph turn: {str(e)}"
        )

    config = {"configurable": {"thread_id": composite_thread_id}}
    state_snapshot = trip_graph.graph.get_state(config)

    return _extract_response_data(
        result=result,
        state_snapshot=state_snapshot,
        user_id=user_id,
        thread_id=thread_id,
        session_mgr=session_mgr,
    )


@router.post("/resume", response_model=ChatResponse)
async def resume_turn(
    req: ResumeRequest,
    session_mgr: SessionManagerDep,
    trip_graph: TripGraphDep,
):
    """
    Resume an interrupted graph execution following Human-in-the-Loop (HITL) review.
    
    - Pass `approved: true` to confirm travel options and generate full itinerary.
    - Pass `approved: false` and `feedback` to revise details and re-run planning.
    """
    user_id = req.user_id
    thread_id = req.thread_id
    session_mgr.ensure_thread_exists(user_id=user_id, thread_id=thread_id)

    composite_thread_id = f"{user_id}::{thread_id}"

    try:
        result = await asyncio.to_thread(
            trip_graph.resume_turn,
            approved=req.approved,
            feedback=req.feedback or "",
            thread_id=composite_thread_id,
        )
    except GraphInterrupt:
        config = {"configurable": {"thread_id": composite_thread_id}}
        state_snapshot = trip_graph.graph.get_state(config)
        result = state_snapshot.values
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred while resuming trip graph: {str(e)}"
        )

    config = {"configurable": {"thread_id": composite_thread_id}}
    state_snapshot = trip_graph.graph.get_state(config)

    return _extract_response_data(
        result=result,
        state_snapshot=state_snapshot,
        user_id=user_id,
        thread_id=thread_id,
        session_mgr=session_mgr,
    )
