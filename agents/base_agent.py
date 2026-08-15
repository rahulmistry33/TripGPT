from abc import ABC, abstractmethod
from typing import Any, List, Optional
from langchain_groq import ChatGroq
from config.config import settings


class BaseAgent(ABC):
    """Abstract base class for all trip planning sub-agents."""

    def __init__(self, model_name: Optional[str] = None, temperature: float = 0.2):
        self.llm: ChatGroq = settings.get_llm(temperature=temperature)

    @abstractmethod
    def run(self, state: dict) -> dict:
        """
        Process the graph state and return state updates.
        Must be implemented by child sub-agents.
        """
        pass
