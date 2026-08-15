import uuid
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional


class SessionManager:
    """
    In-Memory Session & Thread Metadata Manager.
    
    Tracks conversation threads associated with users so that the frontend UI
    can display sidebar conversation lists (threads) per user, create new trip sessions,
    and update trip progress status.
    """

    def __init__(self):
        # Key: composite_key ("user_id::thread_id") -> metadata dict
        self._threads: Dict[str, Dict[str, Any]] = {}

    def _composite_key(self, user_id: str, thread_id: str) -> str:
        return f"{user_id}::{thread_id}"

    def create_thread(self, user_id: str, title: Optional[str] = None) -> str:
        """Create a new trip thread for a given user and return the thread_id."""
        short_id = uuid.uuid4().hex[:8]
        thread_id = f"trip_{short_id}"
        now_iso = datetime.now(timezone.utc).isoformat()
        
        default_title = title or "New Travel Plan"
        comp_key = self._composite_key(user_id, thread_id)
        
        self._threads[comp_key] = {
            "thread_id": thread_id,
            "user_id": user_id,
            "title": default_title,
            "created_at": now_iso,
            "updated_at": now_iso,
            "is_complete": False,
            "destination": None,
            "trip_details": {},
            "missing_fields": [],
        }
        return thread_id

    def ensure_thread_exists(self, user_id: str, thread_id: str) -> str:
        """Ensure thread metadata entry exists for (user_id, thread_id). Creates one if missing."""
        comp_key = self._composite_key(user_id, thread_id)
        if comp_key not in self._threads:
            now_iso = datetime.now(timezone.utc).isoformat()
            self._threads[comp_key] = {
                "thread_id": thread_id,
                "user_id": user_id,
                "title": "Travel Plan",
                "created_at": now_iso,
                "updated_at": now_iso,
                "is_complete": False,
                "destination": None,
                "trip_details": {},
                "missing_fields": [],
            }
        return thread_id

    def list_user_threads(self, user_id: str) -> List[Dict[str, Any]]:
        """Retrieve all threads owned by a specific user, ordered by most recently updated."""
        user_threads = [
            meta for comp_key, meta in self._threads.items()
            if meta["user_id"] == user_id
        ]
        # Sort by updated_at descending
        user_threads.sort(key=lambda x: x["updated_at"], reverse=True)
        return user_threads

    def get_thread(self, user_id: str, thread_id: str) -> Optional[Dict[str, Any]]:
        """Get metadata for a specific user thread."""
        comp_key = self._composite_key(user_id, thread_id)
        return self._threads.get(comp_key)

    def update_thread_state(self, user_id: str, thread_id: str, trip_details: Dict[str, Any]):
        """Update thread metadata with latest trip details and completion state."""
        self.ensure_thread_exists(user_id, thread_id)
        comp_key = self._composite_key(user_id, thread_id)
        meta = self._threads[comp_key]
        
        meta["updated_at"] = datetime.now(timezone.utc).isoformat()
        meta["trip_details"] = trip_details
        meta["is_complete"] = trip_details.get("is_complete", False)
        meta["missing_fields"] = trip_details.get("missing_fields", [])
        
        dest = trip_details.get("destination")
        if dest:
            meta["destination"] = dest
            # Auto-update generic title to a descriptive title if title is generic
            if meta["title"] in ["New Travel Plan", "Travel Plan"]:
                meta["title"] = f"Trip to {dest.title()}"

    def delete_thread(self, user_id: str, thread_id: str) -> bool:
        """Delete a user thread."""
        comp_key = self._composite_key(user_id, thread_id)
        if comp_key in self._threads:
            del self._threads[comp_key]
            return True
        return False
