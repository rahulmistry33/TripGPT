from typing import Dict, Any, List
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage
from agents.base_agent import BaseAgent
from state.trip_state import TripDetails


class PlannerAgent(BaseAgent):
    """
    Planner Agent: Interacts with the user to collect and extract required trip details.
    Required details:
      1. Origin (from where)
      2. Destination (to where)
      3. Start date / Travel period
      4. Number of days
      5. Number of people
      6. Type of trip (luxury, budget, family, adventure, romantic, etc.)
      7. Estimated budget
    """

    REQUIRED_FIELDS = [
        "origin",
        "destination",
        "start_date",
        "num_days",
        "num_people",
        "trip_type",
        "budget",
    ]

    def __init__(self, temperature: float = 0.1):
        super().__init__(temperature=temperature)
        self.extractor_llm = self.llm.with_structured_output(TripDetails)


    def extract_details(self, messages: List[Any], current_details: Dict[str, Any]) -> TripDetails:
        """
        Use structured output to extract trip details from conversation history,
        merging with existing details.
        """
        system_prompt = (
            "You are an expert trip planning assistant. Analyze the conversation history and extract all trip details.\n"
            "Here are the previously saved trip details:\n"
            f"{current_details}\n\n"
            "Extract or update any of the following fields if mentioned by the user:\n"
            "- origin (from where)\n"
            "- destination (to where)\n"
            "- start_date (tentative start date or travel period)\n"
            "- num_days (integer duration in days)\n"
            "- num_people (integer count of travelers)\n"
            "- trip_type (e.g. leisure, luxury, budget, family, adventure, romantic, solo)\n"
            "- budget (estimated budget or pricing tier)\n\n"
            "Be strict: Only mark `is_complete` as True if ALL 7 required fields are non-empty and valid.\n"
            "Otherwise, set `is_complete` to False and list the missing field names in `missing_fields`."
        )

        prompt_messages = [SystemMessage(content=system_prompt)] + messages
        try:
            extracted: TripDetails = self.extractor_llm.invoke(prompt_messages)
        except Exception as e:
            # Fallback if structured output fails on edge case
            extracted = TripDetails()

        # Merge extracted with current_details
        merged_dict = dict(current_details or {})
        for field in self.REQUIRED_FIELDS:
            extracted_val = getattr(extracted, field, None)
            if extracted_val is not None:
                merged_dict[field] = extracted_val

        # Compute missing fields
        missing = [field for field in self.REQUIRED_FIELDS if not merged_dict.get(field)]
        is_complete = len(missing) == 0

        merged_dict["is_complete"] = is_complete
        merged_dict["missing_fields"] = missing

        return TripDetails(**merged_dict)

    def generate_followup_message(self, trip_details: TripDetails, messages: List[Any]) -> str:
        """
        Generate a conversational follow-up question asking for missing fields
        or confirming receipt when complete.
        """
        if trip_details.is_complete:
            return (
                f"Awesome! I have gathered all your trip details:\n"
                f"- **From:** {trip_details.origin} -> **To:** {trip_details.destination}\n"
                f"- **Start Date:** {trip_details.start_date} | **Duration:** {trip_details.num_days} days\n"
                f"- **Travelers:** {trip_details.num_people} person(s) | **Type:** {trip_details.trip_type}\n"
                f"- **Budget:** {trip_details.budget}\n\n"
                f"I am now handing this over to our specialized **Flight**, **Hotel**, and **Itinerary** agents to build your full plan!"
            )

        missing_str = ", ".join([f"**{m.replace('_', ' ')}**" for m in trip_details.missing_fields])
        
        prompt = (
            f"You are a friendly and enthusiastic trip planner AI.\n"
            f"Current known trip details: {trip_details.model_dump(exclude={'missing_fields'})}\n"
            f"Still missing details: {missing_str}\n\n"
            f"Write a polite, engaging response acknowledging what we know so far, "
            f"and asking the user to provide the remaining missing details ({missing_str})."
        )
        
        response = self.llm.invoke([SystemMessage(content=prompt)] + messages[-3:])
        return str(response.content)

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute Planner Agent node logic.
        """
        messages = state.get("messages", [])
        current_details = state.get("trip_details", {})

        updated_details = self.extract_details(messages, current_details)
        details_dict = updated_details.model_dump()

        response_text = self.generate_followup_message(updated_details, messages)

        # Append AI response message to conversation history
        return {
            "trip_details": details_dict,
            "messages": [AIMessage(content=response_text)],
        }
