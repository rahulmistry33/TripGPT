import sys
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Ensure UTF-8 output encoding for Windows terminal
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from api.routers import chat, threads
from api.dependencies import get_trip_graph
from tools.mcp_client import cleanup_mcp


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI Lifespan Context Manager.
    
    Handles application startup and shutdown lifecycle hooks cleanly:
    - Startup: Pre-warms TripGraph singleton and initializes MCP client connection.
    - Shutdown: Triggers cleanup_mcp() to terminate background stdio subprocesses.
    """
    print("\n==========================================================")
    print(" [+] Starting TripGPT FastAPI Application & Pre-Warming MCP ")
    print("==========================================================\n")
    
    # Pre-warm TripGraph singleton
    try:
        get_trip_graph()
        print("[FastAPI Lifespan] TripGraph and MCP server tools initialized successfully!")
    except Exception as e:
        print(f"[FastAPI Lifespan Warning] Pre-warming failed: {e}")

    yield  # Application runs while serving requests

    print("\n==========================================================")
    print(" [-] Shutting down TripGPT FastAPI Application & MCP Processes ")
    print("==========================================================\n")
    cleanup_mcp()


app = FastAPI(
    title="TripGPT API",
    description="Multi-Agent LangGraph Travel Planning System powered by MCP tools",
    version="1.0.0",
    lifespan=lifespan,
)

# Enable CORS for frontend UI integration (React, Next.js, Vite, etc.)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust allowed origins for production deployment
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API Routers
app.include_router(chat.router)
app.include_router(threads.router)


@app.get("/")
async def root():
    """Welcome endpoint for API verification."""
    return {
        "message": "Welcome to TripGPT Multi-Agent Travel Planner API! ✈️🌍",
        "documentation": "/docs",
        "status": "online",
    }
