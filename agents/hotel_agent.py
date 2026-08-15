from typing import Dict, Any
from langchain_core.messages import SystemMessage
from agents.base_agent import BaseAgent
from tools import (
    get_mcp_tool_by_name,
    get_tavily_mcp_tool,
    invoke_mcp_tool,
)


class HotelAgent(BaseAgent):
    """
    Hotel Agent: Researches accommodation options using Tavily Search MCP Server.
    """

    def __init__(self, temperature: float = 0.2):
        super().__init__(temperature=temperature)

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute Hotel Agent node logic over MCP protocol.
        """
        trip_details = state.get("trip_details", {})
        destination = trip_details.get("destination", "N/A")
        num_days = trip_details.get("num_days", 1)
        num_people = trip_details.get("num_people", 1)
        budget = trip_details.get("budget", "mid-range")
        trip_type = trip_details.get("trip_type", "leisure")

        print(f"\n[HotelAgent] Researching stays in {destination} ({budget} budget, {trip_type} trip)...")

        tavily_query = (
            f"best hotels and stays in {destination} for {trip_type} trip "
            f"budget {budget} for {num_people} guests"
        )
        
        tavily_results = ""
        mcp_tavily_tool = get_tavily_mcp_tool()
        try:
            if mcp_tavily_tool:
                print(f"[HotelAgent -> Tavily Remote MCP] Researching stays via '{mcp_tavily_tool.name}'...")
                tavily_results = str(invoke_mcp_tool(mcp_tavily_tool, {"query": tavily_query, "max_results": 3}))
            else:
                tavily_results = "Error: Tavily MCP tool not available."
        except Exception as e:
            tavily_results = f"Hotel search error: {str(e)}"



        # Truncate search results to manage prompt tokens
        truncated_results = tavily_results[:1200]

        prompt = (
            f"You are a specialized Hotel Agent.\n"
            f"Trip Context: Destination={destination}, Duration={num_days} days, "
            f"Travelers={num_people}, Budget Tier={budget}, Type={trip_type}.\n\n"
            f"Search Snippets (from MCP Server):\n{truncated_results}\n\n"
            f"Synthesize 3 distinct hotel/stay options matching this budget and trip style.\n"
            f"For each include: Name, Location, estimated price per night, and key highlights.\n"
            f"Keep output concise (under 300 words)."
        )

        response = self.llm.invoke([SystemMessage(content=prompt)])
        hotel_summary = str(response.content)

        return {"hotel_data": hotel_summary}

