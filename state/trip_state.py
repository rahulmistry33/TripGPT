from typing import Annotated, Optional, List, Dict, Any
from typing_extensions import TypedDict
from pydantic import BaseModel, Field
from langgraph.graph.message import add_messages


class TripDetails(BaseModel):
    """Pydantic model representing extracted trip specifications."""

    origin: Optional[str] = Field(
        default=None,
        description="Origin city or airport (from where the user is traveling).",
    )
    destination: Optional[str] = Field(
        default=None,
        description="Destination city or region (where the user is traveling to).",
    )
    start_date: Optional[str] = Field(
        default=None,
        description="Tentative trip start date or travel period (e.g. '2026-10-01' or 'next month').",
    )
    num_days: Optional[int] = Field(
        default=None,
        description="Duration of the trip in days.",
    )
    num_people: Optional[int] = Field(
        default=None,
        description="Number of travelers / people going on the trip.",
    )
    trip_type: Optional[str] = Field(
        default=None,
        description="Style or type of trip (e.g., leisure, luxury, budget, family, adventure, romantic, solo).",
    )
    budget: Optional[str] = Field(
        default=None,
        description="Estimated budget or price tier (e.g., '$3000 total', '$500/day', 'budget', 'mid-range', 'luxury').",
    )
    is_complete: bool = Field(
        default=False,
        description="Set to True ONLY IF all 7 key fields (origin, destination, start_date, num_days, num_people, trip_type, budget) are known.",
    )
    missing_fields: List[str] = Field(
        default_factory=list,
        description="List of required fields that are still missing or unspecified.",
    )


class TripState(TypedDict, total=False):
    """Graph state definition for the multi-agent trip planning workflow."""

    messages: Annotated[list, add_messages]
    trip_details: Dict[str, Any]
    flight_data: str
    hotel_data: str
    weather_data: str
    itinerary_data: str
    final_response: str

    # Guardrails & HITL state extensions
    guardrail_status: Optional[Dict[str, Any]]
    awaiting_approval: Optional[bool]
    approval_summary: Optional[str]
    approval_response: Optional[Dict[str, Any]]


