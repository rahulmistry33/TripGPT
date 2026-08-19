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

import os
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

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

# Mount static files directory if it exists
static_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "static")
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

# Include API Routers
app.include_router(chat.router)
app.include_router(threads.router)


@app.get("/", response_class=FileResponse)
async def root():
    """Serves the main TripGPT Chatbot UI application."""
    index_path = os.path.join(static_dir, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {
        "message": "Welcome to TripGPT Multi-Agent Travel Planner API! ✈️🌍",
        "documentation": "/docs",
        "status": "online",
    }


@app.get("/api/v1/health")
async def health_check():
    """Health check endpoint for Render monitoring."""
    return {"status": "healthy", "service": "TripGPT"}

