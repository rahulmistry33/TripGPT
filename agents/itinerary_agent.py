from typing import Dict, Any
from langchain_core.messages import SystemMessage, AIMessage
from agents.base_agent import BaseAgent
from tools import get_tavily_mcp_tool, invoke_mcp_tool


class ItineraryAgent(BaseAgent):
    """
    Itinerary Agent: Aggregates flight, hotel, and destination research into a complete day-by-day travel plan.
    """

    def __init__(self, temperature: float = 0.3):
        super().__init__(temperature=temperature)

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute Itinerary Agent node logic.
        """
        trip_details = state.get("trip_details", {})
        flight_data = state.get("flight_data", "No flight info gathered.")
        hotel_data = state.get("hotel_data", "No hotel info gathered.")
        weather_data = state.get("weather_data", "No weather forecast gathered.")

        origin = trip_details.get("origin", "N/A")
        destination = trip_details.get("destination", "N/A")
        start_date = trip_details.get("start_date", "N/A")
        num_days = trip_details.get("num_days", 1)
        num_people = trip_details.get("num_people", 1)
        trip_type = trip_details.get("trip_type", "leisure")
        budget = trip_details.get("budget", "N/A")

        print(f"\n[ItineraryAgent] Researching activities and building {num_days}-day itinerary for {destination}...")

        attractions_query = f"top attractions things to do itinerary in {destination} for {num_days} days"
        mcp_tavily_tool = get_tavily_mcp_tool()
        try:
            if mcp_tavily_tool:
                print(f"[ItineraryAgent -> Tavily Remote MCP] Researching attractions via '{mcp_tavily_tool.name}'...")
                attractions_results = str(invoke_mcp_tool(mcp_tavily_tool, {"query": attractions_query, "max_results": 3}))
            else:
                attractions_results = "Error: Tavily MCP tool not available."
        except Exception as e:
            attractions_results = f"Attractions search error: {str(e)}"

        # Truncate inputs to strictly manage prompt token limits
        truncated_attractions = attractions_results[:1200]
        truncated_flight = flight_data[:800]
        truncated_hotel = hotel_data[:800]
        truncated_weather = weather_data[:800]

        prompt = (
            f"You are the Master Itinerary Agent.\n"
            f"Synthesize all gathered data into a comprehensive day-by-day travel itinerary.\n\n"
            f"### TRIP SPECIFICATIONS:\n"
            f"- From: {origin} -> To: {destination}\n"
            f"- Start Date: {start_date} | Duration: {num_days} Days\n"
            f"- Travelers: {num_people} Person(s) | Trip Style: {trip_type} | Budget: {budget}\n\n"
            f"### FLIGHT DATA:\n{truncated_flight}\n\n"
            f"### HOTEL DATA:\n{truncated_hotel}\n\n"
            f"### WEATHER & CLIMATE ADVISORY:\n{truncated_weather}\n\n"
            f"### DESTINATION ACTIVITIES:\n{truncated_attractions}\n\n"
            f"### REQUIRED STRUCTURE:\n"
            f"1. **Executive Summary**: Overview & selected stays/flights.\n"
            f"2. **Flight & Pickup Details**: Arrival & transfer advice.\n"
            f"3. **Recommended Stays**: Hotel choice highlights.\n"
            f"4. **Weather Forecast & Packing Guide**: Climate expectation, essential packing items.\n"
            f"5. **Day-by-Day Schedule** (Day 1 to Day {num_days}): Morning, Afternoon, Evening activities & dining aligned with weather.\n"
            f"6. **Tips & Budget Overview**: Local transit & spending advice.\n\n"
            f"Use Markdown headers, bullet points, and emojis."
        )


        response = self.llm.invoke([SystemMessage(content=prompt)])
        final_itinerary = str(response.content)

        return {
            "itinerary_data": final_itinerary,
            "final_response": final_itinerary,
            "messages": [AIMessage(content=final_itinerary)],
        }
