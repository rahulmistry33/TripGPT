import sys
import asyncio

# Ensure UTF-8 output encoding for Windows terminal
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from tools.mcp_client import get_mcp_tools, get_mcp_tool_by_name, invoke_mcp_tool



def test_mcp_multi_server():
    print("==================================================")
    print(" [MCP TEST] Testing Multi-Server MCP Setup for trip-gpt")
    print("==================================================\n")

    # 1. Discover tools from all registered MCP Servers
    print("[1] Discovering tools across registered MCP servers...")
    tools = get_mcp_tools()
    tool_names = [t.name for t in tools]
    print(f"[SUCCESS] Discovered {len(tools)} MCP tool(s): {tool_names}\n")

    # 2. Test Open-Source AviationStack MCP Server tool call
    print("[2] Testing Open-Source AviationStack MCP Server...")
    flight_tool = (
        get_mcp_tool_by_name("flight_arrival_departure_schedule")
        or get_mcp_tool_by_name("flights_with_airline")
        or get_mcp_tool_by_name("get_flight_status")
    )
    if not flight_tool:
        for t in tools:
            if "flight" in t.name.lower() or "aviation" in t.name.lower():
                flight_tool = t
                break

    if flight_tool:
        print(f"[FOUND] Utilizing Flight MCP Tool: '{flight_tool.name}'")
        tool_args = {}
        if flight_tool.name == "flights_with_airline":
            tool_args["airline_name"] = "American Airlines"
            tool_args["number_of_flights"] = 2
        elif "dep_iata" in str(flight_tool.args):
            tool_args["dep_iata"] = "JFK"
            tool_args["arr_iata"] = "LHR"
        elif "dep" in str(flight_tool.args):
            tool_args["dep"] = "JFK"
            tool_args["arr"] = "LHR"

        res = invoke_mcp_tool(flight_tool, tool_args)
        print(f"[SUCCESS] Flight MCP Response Snippet:\n{str(res)[:400]}...\n")
    else:
        print("[FAIL] Flight MCP tool not found on AviationStack MCP Server.")


    # 3. Test Official Tavily Remote MCP Server tool call
    print("[3] Testing Official Tavily Search MCP Server...")
    tavily_tool = (
        get_mcp_tool_by_name("tavily_search")
        or get_mcp_tool_by_name("tavily_search_results_json")
        or get_mcp_tool_by_name("search_travel_web")
    )
    if not tavily_tool:
        # Fallback to any tool containing tavily or search
        for t in tools:
            if "tavily" in t.name.lower() or "search" in t.name.lower():
                tavily_tool = t
                break

    if tavily_tool:
        print(f"[FOUND] Utilizing Tavily MCP Tool: '{tavily_tool.name}'")
        res = invoke_mcp_tool(tavily_tool, {
            "query": "top 3 luxury boutique hotels in Tokyo",
            "max_results": 2
        })
        print(f"[SUCCESS] Tavily MCP Response Snippet:\n{str(res)[:400]}...\n")
    else:
        print("[FAIL] Tavily MCP Search tool not found.")


    # 4. Test Custom Weather MCP Server tool call
    print("[4] Testing Custom Weather MCP Server...")
    weather_tool = (
        get_mcp_tool_by_name("get_weather_forecast")
        or get_mcp_tool_by_name("get_current_weather")
    )
    if not weather_tool:
        for t in tools:
            if "weather" in t.name.lower() or "forecast" in t.name.lower():
                weather_tool = t
                break

    if weather_tool:
        print(f"[FOUND] Utilizing Weather MCP Tool: '{weather_tool.name}'")
        res = invoke_mcp_tool(weather_tool, {
            "city": "Tokyo",
            "days": 3
        })
        print(f"[SUCCESS] Weather MCP Response Snippet:\n{str(res)[:400]}...\n")
    else:
        print("[FAIL] Weather MCP tool not found on Weather MCP Server.")

    print("==================================================")
    print(" [MCP TEST] All MCP Multi-Server Tests Completed!")
    print("==================================================")




if __name__ == "__main__":
    try:
        test_mcp_multi_server()
    finally:
        from tools.mcp_client import cleanup_mcp
        cleanup_mcp()
