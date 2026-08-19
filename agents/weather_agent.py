from typing import Dict, Any
from langchain_core.messages import SystemMessage
from agents.base_agent import BaseAgent
from tools import (
    get_weather_mcp_tool,
    get_mcp_tool_by_name,
    invoke_mcp_tool,
)


class WeatherAgent(BaseAgent):
    """
    Weather Agent: Fetches real-time weather forecasts and current conditions for travel destinations
    using the custom Weather MCP Server over stdio.
    """

    def __init__(self, temperature: float = 0.2):
        super().__init__(temperature=temperature)

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute Weather Agent node logic over Weather MCP Server protocol.
        """
        trip_details = state.get("trip_details", {})
        destination = trip_details.get("destination", "N/A")
        start_date = trip_details.get("start_date", "N/A")
        num_days = trip_details.get("num_days", 5) or 5

        print(f"\n[WeatherAgent] Fetching weather forecast for {destination} ({num_days} days)...")

        weather_results = ""
        mcp_weather_tool = (
            get_mcp_tool_by_name("get_weather_forecast")
            or get_weather_mcp_tool()
        )

        try:
            if mcp_weather_tool:
                print(f"[WeatherAgent -> Weather MCP Server] Invoking '{mcp_weather_tool.name}'...")
                tool_args = {"city": destination, "days": int(num_days)}
                weather_results = str(invoke_mcp_tool(mcp_weather_tool, tool_args))
            else:
                weather_results = "Error: Weather MCP tool not available."
        except Exception as e:
            weather_results = f"Weather search error: {str(e)}"

        # Truncate raw search results to manage prompt tokens
        truncated_weather = weather_results[:1200]

        prompt = (
            f"You are a specialized Weather & Climate Travel Agent.\n"
            f"Trip Context: Destination={destination}, Start Date={start_date}, Duration={num_days} days.\n\n"
            f"Live Weather Data (from Weather MCP Server):\n{truncated_weather}\n\n"
            f"FORMAT INSTRUCTIONS:\n"
            f"- Do NOT use top-level markdown headings (# or ##).\n"
            f"- Output 2-3 concise bullet points starting with 🌤️ and 🎒.\n"
            f"- Include: Expected temperature range/climate, essential packing advice, and best time of day for outdoor exploration.\n"
            f"- Keep total response under 100 words."
        )

        response = self.llm.invoke([SystemMessage(content=prompt)])
        weather_summary = str(response.content)

        return {"weather_data": weather_summary}

