import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

# Load environment variables
load_dotenv()


class Settings:
    """Application settings and LLM configuration."""

    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    TAVILY_API_KEY: str = os.getenv("TAVILY_API_KEY", "")
    AVIATIONSTACK_API_KEY: str = os.getenv("AVIATIONSTACK_API_KEY", "")
    DEFAULT_MODEL_NAME: str = os.getenv("MODEL_NAME", "llama-3.3-70b-versatile")

    @classmethod
    def get_llm(cls, temperature: float = 0.2) -> ChatGroq:
        """Initialize and return a ChatGroq LLM instance."""
        if not cls.GROQ_API_KEY:
            raise ValueError(
                "GROQ_API_KEY is not set. Please set GROQ_API_KEY in your .env file."
            )

        return ChatGroq(
            model_name=cls.DEFAULT_MODEL_NAME,
            temperature=temperature,
            api_key=cls.GROQ_API_KEY,
        )


settings = Settings()
