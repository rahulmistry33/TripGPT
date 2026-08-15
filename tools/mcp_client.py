import os
import sys
import atexit
import asyncio
from typing import List, Dict, Any, Optional
from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient

# Global client and tools cache for efficiency
_mcp_client: Optional[MultiServerMCPClient] = None
_mcp_tools_cache: Optional[List[BaseTool]] = None


def get_mcp_server_config() -> Dict[str, Dict[str, Any]]:
    """
    Returns the MultiServerMCPClient connection configuration dictionary.
    - Tavily Search: Official Remote MCP Server via streamable_http (https://mcp.tavily.com/mcp/?tavilyApiKey=...)
    - AviationStack: Open-Source MCP Server (aviationstack-mcp) executed via python -m aviationstack_mcp over stdio
    """
    python_exe = sys.executable
    tavily_api_key = os.getenv("TAVILY_API_KEY", "")
    aviation_api_key = os.getenv("AVIATIONSTACK_API_KEY", "") or os.getenv("AVIATION_STACK_API_KEY", "")

    # Environment variables dict for AviationStack MCP process
    aviation_env = dict(os.environ)
    if aviation_api_key:
        aviation_env["AVIATION_STACK_API_KEY"] = aviation_api_key
        aviation_env["AVIATIONSTACK_API_KEY"] = aviation_api_key

    return {
        "tavily": {
            "transport": "streamable_http",
            "url": f"https://mcp.tavily.com/mcp/?tavilyApiKey={tavily_api_key}",
        },
        "aviationstack": {
            "command": python_exe,
            "args": ["-m", "aviationstack_mcp"],
            "transport": "stdio",
            "env": aviation_env,
        },
        "weather": {
            "command": python_exe,
            "args": ["-m", "mcp_servers.weather_server"],
            "transport": "stdio",
        },
    }







async def async_get_mcp_tools() -> List[BaseTool]:
    """
    Asynchronously initializes MultiServerMCPClient and fetches all MCP tools from registered servers.
    """
    global _mcp_client, _mcp_tools_cache
    if _mcp_tools_cache is None:
        config = get_mcp_server_config()
        _mcp_client = MultiServerMCPClient(config)
        _mcp_tools_cache = await _mcp_client.get_tools()
    return _mcp_tools_cache


def get_mcp_tools() -> List[BaseTool]:
    """
    Synchronous helper to retrieve all MCP tools.
    Supports running event loops via nest_asyncio if needed.
    """
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import nest_asyncio
            nest_asyncio.apply()
            return loop.run_until_complete(async_get_mcp_tools())
        else:
            return loop.run_until_complete(async_get_mcp_tools())
    except RuntimeError:
        return asyncio.run(async_get_mcp_tools())


async def async_cleanup_mcp():
    """Reset MCP client and cached tools."""
    global _mcp_client, _mcp_tools_cache
    _mcp_client = None
    _mcp_tools_cache = None


def cleanup_mcp():
    """Synchronous cleanup wrapper for MCP client resources."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import nest_asyncio
            nest_asyncio.apply()
            loop.run_until_complete(async_cleanup_mcp())
        else:
            loop.run_until_complete(async_cleanup_mcp())
    except RuntimeError:
        try:
            asyncio.run(async_cleanup_mcp())
        except RuntimeError:
            pass  # Event loop already closed at interpreter shutdown


# Best-effort cleanup on program exit (terminates MCP child processes)
atexit.register(cleanup_mcp)


def get_mcp_tool_by_name(tool_name: str) -> Optional[BaseTool]:
    """
    Helper to fetch a specific MCP tool by its name (e.g. 'get_flight_status' or 'tavily_search').
    """
    tools = get_mcp_tools()
    for tool in tools:
        if tool.name == tool_name:
            return tool
    return None


def get_tavily_mcp_tool() -> Optional[BaseTool]:
    """
    Helper to fetch Tavily MCP web search tool from registered MCP servers.
    Prioritizes tools with 'tavily' in the name, falls back to 'search'.
    """
    tools = get_mcp_tools()
    # First pass: prefer tools explicitly named 'tavily'
    for tool in tools:
        if "tavily" in tool.name.lower():
            return tool
    # Second pass: fallback to any tool with 'search' in name
    for tool in tools:
        if "search" in tool.name.lower():
            return tool
    return None


def get_flight_mcp_tool(preferred_name: Optional[str] = None) -> Optional[BaseTool]:
    """
    Helper to fetch Flight MCP tool from registered MCP servers (e.g. aviationstack-mcp).
    Optionally match a specific tool name, otherwise searches for 'flight' then 'aviation'.
    """
    tools = get_mcp_tools()
    # Exact match if a preferred name is given
    if preferred_name:
        for tool in tools:
            if tool.name == preferred_name:
                return tool
    # First pass: prefer tools with 'flight' in name
    for tool in tools:
        if "flight" in tool.name.lower():
            return tool
    # Second pass: fallback to 'aviation'
    for tool in tools:
        if "aviation" in tool.name.lower():
            return tool
    return None


def get_weather_mcp_tool(preferred_name: Optional[str] = None) -> Optional[BaseTool]:

    """
    Helper to fetch Weather MCP tool from registered MCP servers (e.g. weather_server).
    Optionally match a specific tool name, otherwise searches for 'weather' then 'forecast'.
    """
    tools = get_mcp_tools()
    if preferred_name:
        for tool in tools:
            if tool.name == preferred_name:
                return tool
    for tool in tools:
        if "weather" in tool.name.lower():
            return tool
    for tool in tools:
        if "forecast" in tool.name.lower():
            return tool
    return None


def invoke_mcp_tool(tool: BaseTool, tool_input: Dict[str, Any]) -> Any:

    """
    Invokes an MCP tool safely handling async/sync execution contexts (via tool.ainvoke).
    """
    async def _runner():
        return await tool.ainvoke(tool_input)

    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import nest_asyncio
            nest_asyncio.apply()
            return loop.run_until_complete(_runner())
        else:
            return loop.run_until_complete(_runner())
    except RuntimeError:
        return asyncio.run(_runner())

