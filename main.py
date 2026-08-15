import sys
from dotenv import load_dotenv

# Ensure UTF-8 output encoding for Windows terminal
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from graph.trip_graph import TripGraph
from tools.mcp_client import cleanup_mcp

# Load environment variables
load_dotenv()


def run_interactive():
    """
    Run an interactive multi-turn CLI session with the TripGPT multi-agent system.
    """
    print("\n" + "=" * 75)
    print("      🚀 Welcome to TripGPT - Multi-Agent LangGraph Planner 🚀      ")
    print("=" * 75)
    print("Tell me about your travel plans! (Type 'exit' or 'quit' to end session)\n")

    # Initialize graph
    trip_graph = TripGraph()
    thread_id = "user_session_1"

    while True:
        try:
            user_input = input("You: ").strip()
            if not user_input:
                continue

            if user_input.lower() in ["exit", "quit"]:
                print("\nThank you for using TripGPT! Happy travels! ✈️🌍")
                break

            print("\nTripGPT thinking...")
            result = trip_graph.run_turn(user_input=user_input, thread_id=thread_id)

            # Retrieve latest AI message from state
            messages = result.get("messages", [])
            if messages:
                latest_msg = messages[-1]
                print(f"\nTripGPT:\n{latest_msg.content}\n")
                print("-" * 75)

            # Display current extracted details status
            details = result.get("trip_details", {})
            if details:
                is_comp = details.get("is_complete", False)
                missing = details.get("missing_fields", [])
                print(f"[State Update] Complete: {is_comp} | Missing: {missing if missing else 'None'}\n")

        except KeyboardInterrupt:
            print("\nSession interrupted. Exiting...")
            break
        except Exception as e:
            print(f"\nAn error occurred during execution: {str(e)}\n")

    # Cleanup MCP server connections and child processes
    print("[Cleanup] Shutting down MCP connections...")
    cleanup_mcp()


if __name__ == "__main__":
    run_interactive()
