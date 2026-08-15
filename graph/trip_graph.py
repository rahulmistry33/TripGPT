from typing import Dict, Any, List, Union
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import RetryPolicy, interrupt, Command
from langchain_core.messages import HumanMessage, AIMessage

from state.trip_state import TripState
from agents import (
    PlannerAgent,
    FlightAgent,
    HotelAgent,
    WeatherAgent,
    ItineraryAgent,
    GuardrailAgent,
)


class TripGraph:
    """
    LangGraph Workflow Builder & Compiler for Multi-Agent Trip Planning
    with Guardrails (input/topic/parameter/output validation) and
    Human-in-the-Loop (HITL state interrupt & review resume).
    """

    def __init__(self):
        # Instantiate agents
        self.guardrail_agent = GuardrailAgent()
        self.planner_agent = PlannerAgent()
        self.flight_agent = FlightAgent()
        self.hotel_agent = HotelAgent()
        self.weather_agent = WeatherAgent()
        self.itinerary_agent = ItineraryAgent()

        # Checkpointer for multi-turn state persistence
        self.memory = MemorySaver()

        # Build and compile graph
        self.graph = self._build_graph()

    def _guardrail_node(self, state: TripState) -> Dict[str, Any]:
        """Node execution wrapper for Input Guardrail Agent."""
        return self.guardrail_agent.run(state)

    def _planner_node(self, state: TripState) -> Dict[str, Any]:
        """Node execution wrapper for Planner Agent."""
        return self.planner_agent.run(state)

    def _flight_node(self, state: TripState) -> Dict[str, Any]:
        """Node execution wrapper for Flight Agent."""
        return self.flight_agent.run(state)

    def _hotel_node(self, state: TripState) -> Dict[str, Any]:
        """Node execution wrapper for Hotel Agent."""
        return self.hotel_agent.run(state)

    def _weather_node(self, state: TripState) -> Dict[str, Any]:
        """Node execution wrapper for Weather Agent."""
        return self.weather_agent.run(state)

    def _approval_node(self, state: TripState) -> Dict[str, Any]:
        """
        Human-in-the-Loop (HITL) Approval Node.
        Formulates trip options summary and halts execution using LangGraph's native interrupt().
        Resumes when Command(resume=...) is passed by the client/user.
        """
        details = state.get("trip_details", {})
        flight_preview = state.get("flight_data", "No flight data")[:200]
        hotel_preview = state.get("hotel_data", "No hotel data")[:200]
        weather_preview = state.get("weather_data", "No weather data")[:150]

        summary = (
            f"📋 **Human Approval Required**: Flight, Hotel, & Weather research complete.\n"
            f"- **Destination:** {details.get('destination', 'N/A')} ({details.get('num_days', 'N/A')} days, {details.get('num_people', 'N/A')} people)\n"
            f"- **Budget:** {details.get('budget', 'N/A')} | **Type:** {details.get('trip_type', 'N/A')}\n"
            f"- **Flight Options:** {flight_preview}...\n"
            f"- **Hotel Options:** {hotel_preview}...\n"
            f"- **Weather Summary:** {weather_preview}...\n\n"
            f"Please approve to finalize your full day-by-day itinerary, or provide feedback for revisions."
        )

        # Trigger native LangGraph interrupt
        interrupt_payload = {
            "action": "human_approval_required",
            "summary": summary,
            "trip_details": details,
        }

        # Interrupt execution and wait for Command(resume=...)
        resume_val = interrupt(interrupt_payload)

        # Handle resumed payload
        approved = True
        feedback = ""
        if isinstance(resume_val, dict):
            approved = resume_val.get("approved", True)
            feedback = resume_val.get("feedback", "")

        update = {
            "awaiting_approval": False,
            "approval_summary": summary,
            "approval_response": {"approved": approved, "feedback": feedback},
        }

        if not approved and feedback:
            update["messages"] = [HumanMessage(content=f"User feedback for itinerary revision: {feedback}")]

        return update

    def _itinerary_node(self, state: TripState) -> Dict[str, Any]:
        """Node execution wrapper for Itinerary Agent with Output Guardrail compliance."""
        res = self.itinerary_agent.run(state)
        
        # Apply output guardrail to final message
        messages = res.get("messages", [])
        if messages:
            last_msg = messages[-1]
            raw_text = getattr(last_msg, "content", str(last_msg))
            guarded_text = self.guardrail_agent.validate_output(raw_text)
            res["messages"] = [AIMessage(content=guarded_text)]
            res["final_response"] = guarded_text

        return res

    def _route_after_guardrail(self, state: TripState) -> str:
        """
        Guardrail Router:
        If input guardrail fails, halt graph immediately (route to END).
        Otherwise proceed to planner agent.
        """
        guardrail_status = state.get("guardrail_status", {})
        passed = guardrail_status.get("passed", True)

        if not passed:
            print("\n[Graph Router] Input Guardrail Failed -> Short-circuiting execution to END.")
            return END

        return "planner"

    def _route_after_planner(self, state: TripState) -> Union[List[str], str]:
        """
        Conditional Router:
        If trip details are incomplete, pause execution (route to END) to wait for next user input.
        If complete, fan out in parallel to flight, hotel, and weather nodes!
        """
        trip_details = state.get("trip_details", {})
        is_complete = trip_details.get("is_complete", False)

        if not is_complete:
            print("\n[Graph Router] Trip details incomplete -> Awaiting user input.")
            return END
        else:
            print("\n[Graph Router] All trip details complete! Fanning out to Flight, Hotel, and Weather agents...")
            return ["flight", "hotel", "weather"]

    def _route_after_approval(self, state: TripState) -> str:
        """
        HITL Router:
        If human approved, proceed to itinerary node to build final plan.
        If human rejected with feedback, route back to planner to revise trip specifications.
        """
        approval_response = state.get("approval_response", {})
        approved = approval_response.get("approved", True)

        if approved:
            print("\n[Graph Router] Human approved options -> Routing to Itinerary Agent!")
            return "itinerary"
        else:
            print("\n[Graph Router] Human requested revisions -> Routing back to Planner Agent with feedback!")
            return "planner"

    def _build_graph(self):
        """Constructs and compiles the LangGraph StateGraph."""
        builder = StateGraph(TripState)

        retry_policy = RetryPolicy(
            max_attempts=3,
            initial_interval=3.0,
            backoff_factor=2.0,
        )

        # 1. Add Agent Nodes with retry policies
        builder.add_node("guardrail", self._guardrail_node, retry_policy=retry_policy)
        builder.add_node("planner", self._planner_node, retry_policy=retry_policy)
        builder.add_node("flight", self._flight_node, retry_policy=retry_policy)
        builder.add_node("hotel", self._hotel_node, retry_policy=retry_policy)
        builder.add_node("weather", self._weather_node, retry_policy=retry_policy)
        builder.add_node("approval", self._approval_node, retry_policy=retry_policy)
        builder.add_node("itinerary", self._itinerary_node, retry_policy=retry_policy)

        # 2. Add Edges & Conditional Routing
        builder.add_edge(START, "guardrail")

        builder.add_conditional_edges(
            "guardrail",
            self._route_after_guardrail,
        )

        builder.add_conditional_edges(
            "planner",
            self._route_after_planner,
        )

        # Parallel fan-in to approval (HITL interrupt point)
        builder.add_edge("flight", "approval")
        builder.add_edge("hotel", "approval")
        builder.add_edge("weather", "approval")

        builder.add_conditional_edges(
            "approval",
            self._route_after_approval,
        )

        builder.add_edge("itinerary", END)

        # 3. Compile with Checkpointer memory
        return builder.compile(checkpointer=self.memory)

    def run_turn(self, user_input: str, thread_id: str = "default_session") -> Dict[str, Any]:
        """
        Execute a single conversation turn against the state graph.
        """
        config = {"configurable": {"thread_id": thread_id}}
        input_state = {
            "messages": [HumanMessage(content=user_input)],
        }
        
        # Invoke state graph
        output_state = self.graph.invoke(input_state, config=config)
        return output_state

    def resume_turn(self, approved: bool, feedback: str = "", thread_id: str = "default_session") -> Dict[str, Any]:
        """
        Resume an interrupted graph execution following human approval/feedback.
        """
        config = {"configurable": {"thread_id": thread_id}}
        resume_payload = {"approved": approved, "feedback": feedback}
        
        output_state = self.graph.invoke(Command(resume=resume_payload), config=config)
        return output_state
