from typing import Annotated
from fastapi import Depends

from graph.trip_graph import TripGraph
from api.session_manager import SessionManager

# Global singleton instances
_trip_graph_instance: TripGraph | None = None
_session_manager_instance: SessionManager | None = None


def get_trip_graph() -> TripGraph:
    """Dependency provider for singleton TripGraph instance."""
    global _trip_graph_instance
    if _trip_graph_instance is None:
        print("[Dependencies] Initializing global TripGraph instance...")
        _trip_graph_instance = TripGraph()
    return _trip_graph_instance


def get_session_manager() -> SessionManager:
    """Dependency provider for singleton SessionManager instance."""
    global _session_manager_instance
    if _session_manager_instance is None:
        print("[Dependencies] Initializing global SessionManager instance...")
        _session_manager_instance = SessionManager()
    return _session_manager_instance


# Type Aliases for FastAPI Dependency Injection
TripGraphDep = Annotated[TripGraph, Depends(get_trip_graph)]
SessionManagerDep = Annotated[SessionManager, Depends(get_session_manager)]
