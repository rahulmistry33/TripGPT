from typing import Dict, Any
from langchain_core.messages import SystemMessage
from agents.base_agent import BaseAgent
from tools import (
    get_mcp_tool_by_name,
    get_tavily_mcp_tool,
    get_flight_mcp_tool,
    invoke_mcp_tool,
)


# Common city name to IATA airport code mapping for robust resolution
_CITY_TO_IATA = {
    "new york": "JFK", "nyc": "JFK", "manhattan": "JFK",
    "los angeles": "LAX", "la": "LAX",
    "san francisco": "SFO",
    "chicago": "ORD",
    "miami": "MIA",
    "london": "LHR",
    "paris": "CDG",
    "tokyo": "NRT", "narita": "NRT",
    "dubai": "DXB",
    "singapore": "SIN",
    "hong kong": "HKG",
    "bangkok": "BKK",
    "sydney": "SYD",
    "toronto": "YYZ",
    "frankfurt": "FRA",
    "amsterdam": "AMS",
    "rome": "FCO",
    "istanbul": "IST",
    "cairo": "CAI",
    "seoul": "ICN", "incheon": "ICN",
    "beijing": "PEK",
    "shanghai": "PVG",
    "mumbai": "BOM", "bombay": "BOM",
    "delhi": "DEL", "new delhi": "DEL",
    "bangalore": "BLR", "bengaluru": "BLR",
    "hyderabad": "HYD",
    "chennai": "MAA",
    "kolkata": "CCU", "calcutta": "CCU",
    "goa": "GOI",
    "jaipur": "JAI",
    "ahmedabad": "AMD",
    "pune": "PNQ",
    "kochi": "COK", "cochin": "COK",
    "kuala lumpur": "KUL",
    "bali": "DPS", "denpasar": "DPS",
    "doha": "DOH",
    "abu dhabi": "AUH",
    "riyadh": "RUH",
}


def _resolve_iata(location: str) -> str:
    """
    Resolve a city name or IATA code string to a 3-letter IATA airport code.
    Returns the original string if no mapping is found (lets the API handle it).
    """
    if not location:
        return location
    loc = location.strip()
    # Already a 3-letter IATA code
    if len(loc) == 3 and loc.isalpha():
        return loc.upper()
    # Lookup in city mapping
    mapped = _CITY_TO_IATA.get(loc.lower())
    if mapped:
        return mapped
    # Return original as-is (API may still handle city names)
    return loc


class FlightAgent(BaseAgent):
    """
    Flight Agent: Researches flight options using AviationStack Flight MCP Server and Tavily Search MCP Server.
    """

    def __init__(self, temperature: float = 0.2):
        super().__init__(temperature=temperature)

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute Flight Agent node logic over MCP protocol.
        """
        trip_details = state.get("trip_details", {})
        origin = trip_details.get("origin", "N/A")
        destination = trip_details.get("destination", "N/A")
        start_date = trip_details.get("start_date", "N/A")
        num_people = trip_details.get("num_people", 1)

        print(f"\n[FlightAgent] Searching flights from {origin} to {destination} on {start_date}...")

        # 1. Open-Source AviationStack flight search via MCP
        aviation_results = ""
        mcp_flight_tool = get_flight_mcp_tool()
        try:
            dep_code = _resolve_iata(origin)
            arr_code = _resolve_iata(destination)

            if mcp_flight_tool:
                tool_args = {}
                if "dep_iata" in str(mcp_flight_tool.args):
                    tool_args["dep_iata"] = dep_code or origin
                    tool_args["arr_iata"] = arr_code or destination
                elif "departure_airport" in str(mcp_flight_tool.args):
                    tool_args["departure_airport"] = origin
                    tool_args["arrival_airport"] = destination

                print(f"[FlightAgent -> AviationStack MCP] Invoking '{mcp_flight_tool.name}' on Open-Source MCP Server...")
                aviation_results = str(invoke_mcp_tool(mcp_flight_tool, tool_args))
            else:
                aviation_results = "Error: Flight MCP tool not available."
        except Exception as e:
            aviation_results = f"AviationStack search error: {str(e)}"


        # 2. Web search via Tavily Remote MCP Server
        tavily_query = f"flights from {origin} to {destination} starting around {start_date} airlines schedules"
        tavily_results = ""
        mcp_tavily_tool = get_tavily_mcp_tool()
        try:
            if mcp_tavily_tool:
                print(f"[FlightAgent -> Tavily Remote MCP] Invoking '{mcp_tavily_tool.name}' on Tavily MCP Server...")
                tavily_results = str(invoke_mcp_tool(mcp_tavily_tool, {"query": tavily_query, "max_results": 3}))
            else:
                tavily_results = "Error: Tavily MCP tool not available."
        except Exception as e:
            tavily_results = f"Tavily search error: {str(e)}"


        
        # Truncate raw search results to prevent token rate limits
        truncated_tavily = tavily_results[:1200]
        truncated_aviation = aviation_results[:1200]

        prompt = (
            f"You are a specialized Flight Agent.\n"
            f"Trip context: {origin} to {destination}, starting {start_date}, for {num_people} passenger(s).\n\n"
            f"Flight Data Snippets (from MCP Server):\n{truncated_aviation}\n\n"
            f"Search Snippets (from MCP Server):\n{truncated_tavily}\n\n"
            f"FORMAT INSTRUCTIONS:\n"
            f"- Do NOT use markdown tables or headings (# or ##).\n"
            f"- Output 2-3 realistic flight options as clean bullet points starting with 🛫.\n"
            f"- For each option include: Airline, departure/arrival times, estimated round-trip price, and 1 short transport tip.\n"
            f"- Keep total response under 150 words."
        )

        response = self.llm.invoke([SystemMessage(content=prompt)])
        flight_summary = str(response.content)

        return {"flight_data": flight_summary}


