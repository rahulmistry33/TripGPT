# Comprehensive Project Context & Progress: AI Trip Agent (`trip-gpt`)

## 1. Project Overview & Environment
- **Project Name:** `trip-gpt`
- **Frameworks & Libraries:** Python 3.13, `fastapi`, `uvicorn`, `langchain`, `langgraph`, `langchain-groq`, `pydantic`, `python-dotenv`, `mcp`, `langchain-mcp-adapters`, `aviationstack-mcp`, `nest_asyncio`, `requests`.
- **Package Runner & Environment:** `uv` (Fast Python package manager & runner) + Docker Containerization.
- **Architecture Standard:** 100% Pure Model Context Protocol (MCP) Multi-Server Hybrid Architecture (Remote HTTP + Local Stdio MCP Subprocesses) + Production FastAPI Web Server Layer + LangGraph Guardrails & HITL Subsystems.
- **Production Cloud Target:** Render Web Service (Docker Environment).
- **Configured API Keys (`.env`):**
  - `TAVILY_API_KEY`: Web search & travel destination research via Tavily Remote MCP Server (`streamable_http`).
  - `AVIATIONSTACK_API_KEY` / `AVIATION_STACK_API_KEY`: Real-time flight status, schedules, and airport lookup via AviationStack Stdio MCP Server.
  - `GROQ_API_KEY`: LLM inference (`llama-3.3-70b-versatile`).

---

## 2. Project Directory Structure
```
trip-gpt/
├── .env                        # Environment variables & API keys
├── .env.example                # Template for required API keys
├── .gitignore                  # Git ignore rules (pycache, env, venv, caches)
├── .dockerignore               # Docker build ignore rules
├── Dockerfile                  # Production multi-stage Docker build recipe using Astral UV base image
├── docker-compose.yml          # Local multi-container development configuration
├── render.yaml                 # Render Infrastructure-as-Code (IaC) deployment blueprint
├── pyproject.toml              # Project dependencies & package specs (includes aviationstack-mcp)
├── uv.lock                     # UV dependency lock file
├── requirements.txt            # Package requirements list
├── README.md                   # Complete repository documentation & uv quickstart
├── PROJECT_SUMMARY.md          # Comprehensive project summary & progress log
├── main.py                     # Interactive CLI runner with multi-turn memory & MCP cleanup
├── run_server.py               # Production launcher for FastAPI Uvicorn web server (dynamic PORT binding)
├── generate_openapi.py         # Script to export openapi.json for Postman import
├── openapi.json                # Exported OpenAPI 3.0 specification for 1-click Postman import
├── test_fastapi_app.py         # FastAPI endpoint, Guardrails, & HITL verification test suite
├── test_mcp_servers.py         # Multi-server MCP verification script (19 tools across 3 servers)
├── api/                        # Production FastAPI Web Server Subsystem
│   ├── __init__.py             # Package init
│   ├── main.py                 # FastAPI application, CORS middleware, & Lifespan hooks
│   ├── models.py               # Pydantic V2 Request & Response schemas (Chat, Resume, Threads)
│   ├── session_manager.py      # Multi-tenant user session & thread metadata manager
│   ├── dependencies.py         # FastAPI dependency injection (TripGraph & SessionManager singletons)
│   └── routers/                # API Route Handlers
│       ├── __init__.py         # Package init
│       ├── chat.py             # POST /api/v1/chat & POST /api/v1/chat/resume
│       └── threads.py          # CRUD endpoints for user conversation threads (UI sidebar support)
├── mcp_servers/                # Custom Local Stdio MCP Servers Package
│   ├── __init__.py             # Package init
│   └── weather_server.py       # Custom FastMCP Stdio Weather Server (Open-Meteo REST API)
├── config/
│   └── config.py               # Settings & ChatGroq LLM initialization
├── state/
│   └── trip_state.py           # TripDetails model & TripState TypedDict (with Guardrails & HITL state fields)
├── tools/                      # Pure MCP Client Package
│   ├── __init__.py             # MCP helper exports (get_weather_mcp_tool, get_flight_mcp_tool, etc.)
│   └── mcp_client.py           # MultiServerMCPClient orchestration & Singleton tool caching
├── agents/                     # Modular Sub-Agent Classes
│   ├── __init__.py             # Agent exports
│   ├── base_agent.py           # Abstract BaseAgent class
│   ├── guardrail_agent.py      # GuardrailAgent (input/output validation, topic safety, parameter bounds)
│   ├── planner_agent.py        # PlannerAgent (structured extraction & follow-ups)
│   ├── flight_agent.py         # FlightAgent (100% MCP-based with IATA city resolver)
│   ├── hotel_agent.py          # HotelAgent (100% MCP-based via Tavily MCP)
│   ├── weather_agent.py        # WeatherAgent (100% MCP-based via custom Weather MCP server)
│   └── itinerary_agent.py      # ItineraryAgent (aggregates flight, hotel, weather & activities)
└── graph/                      # LangGraph Architecture
    ├── __init__.py             # Graph exports
    └── trip_graph.py           # TripGraph (Guardrails entry point, 3-way parallel fan-out, HITL interrupt/resume, MemorySaver)
```

---

## 3. Key Architectural Components & Technical Insights

### A. Guardrails Architecture (`agents/guardrail_agent.py`)
- **Input Guardrail (`input_guardrail` node)**: Evaluates prompts immediately after `START`. Rejects non-travel prompts (code generation, math, hacking, prompt injections) and enforces numeric boundary constraints (`1 <= num_days <= 90`, `1 <= num_people <= 50`, `origin != destination`).
- **Short-Circuit Router (`_route_after_guardrail`)**: Immediately halts execution (routes to `END`) if input guardrails fail, returning a formatted warning message without wasting LLM or tool tokens.
- **Output Guardrail (`output_guardrail`)**: Appends mandatory rate/weather volatility disclaimers to final itineraries.

### B. Human-in-the-Loop (HITL) Architecture (`graph/trip_graph.py` & `api/routers/chat.py`)
- **Native State Interrupt (`approval` node)**: Pauses graph execution using LangGraph's native `interrupt()` function after sub-agents (`flight`, `hotel`, `weather`) finish research.
- **State Persistence**: Saves conversation state to `MemorySaver` checkpointer and exposes `awaiting_approval: true` with a preliminary travel options proposal summary.
- **Resume Endpoint (`POST /api/v1/chat/resume`)**:
  - Resumes execution via LangGraph's native `Command(resume=...)`.
  - If approved (`approved: true`), proceeds to `itinerary_agent` for full day-by-day plan generation.
  - If rejected (`approved: false`), routes back to `planner_agent` with user revision feedback.

### C. FastAPI Application Layer & Multi-User Thread Scoping (`api/`)
- **Multi-Tenant User & Thread Architecture (`user_id` + `thread_id`)**: Every trip session is isolated per composite key (`user_id::thread_id`).
- **Endpoints**:
  - `POST /api/v1/chat`: Initiates or continues chat turns with guardrails and HITL interrupts.
  - `POST /api/v1/chat/resume`: Resumes paused HITL execution.
  - `GET /api/v1/users/{user_id}/threads`: Sidebar conversation thread metadata list.
  - `GET /api/v1/users/{user_id}/threads/{thread_id}`: Full message history and trip details.
  - `DELETE /api/v1/users/{user_id}/threads/{thread_id}`: Deletes user conversation thread.

### D. Multi-Server MCP Client & Singleton Caching (`tools/mcp_client.py`)
- 19 total MCP tools across Tavily Remote MCP (`streamable_http`), AviationStack Stdio MCP (`aviationstack-mcp`), and custom Weather Stdio MCP (`mcp_servers.weather_server`).

### E. Containerization & Cloud Deployment Architecture
- **Astral UV Docker Container (`Dockerfile`)**: Optimized multi-stage build leveraging Docker layer caching by installing locked dependencies (`uv sync --frozen --no-dev`) prior to copying project source code.
- **Render Dynamic Port Binding**: `run_server.py` listens to host `0.0.0.0` and dynamically reads the `PORT` environment variable provided by Render.
- **Render Blueprint (`render.yaml`)**: Automated 1-click cloud service provisioning.

---

## 4. Recent DevOps & Deployment Progress

1. **Docker & Containerization Setup:**
   - Authored production-ready `Dockerfile` using `ghcr.io/astral-sh/uv:python3.13-bookworm-slim`.
   - Created `docker-compose.yml` for seamless local multi-container testing (`docker compose up --build`).
   - Created `.dockerignore` to prevent staging temporary files and bytecodes into container context.

2. **Git Cache Cleaning (`__pycache__` Untracking):**
   - Identified tracked `__pycache__` bytecode files in Git index due to early commits prior to `.gitignore` creation.
   - Purged all 29 cached `.pyc` files using `git rm --cached` and committed `chore: untrack pycache bytecode files from git`.

3. **Render Cloud Deployment & Stdio Subprocess Bug Resolution:**
   - Deployed the application to Render Web Services via Docker runtime.
   - Diagnosed Render runtime error (`No module named aviationstack_mcp`) caused by missing package declaration in container `pyproject.toml`.
   - Executed `uv add aviationstack-mcp`, synchronizing `pyproject.toml`, `uv.lock`, and `requirements.txt`.
   - Pushed dependency updates to GitHub master branch, triggering an automated green build on Render.

---

## 5. Verification & Live Status
- **FastAPI Test Suite (`test_fastapi_app.py`)**: 100% PASS across:
  - Root status and thread CRUD.
  - Input Guardrail rejection of non-travel/harmful queries.
  - Multi-agent parallel research & HITL state interrupt (`awaiting_approval: true`).
  - HITL resume (`POST /api/v1/chat/resume`) -> final itinerary generation with Output Guardrails.
- **Local Docker Suite (`docker-compose.yml`)**: 100% PASS on port `8000`.
- **Production Render Web Service**: Live & fully operational with auto-deploy on push.
- **Live OpenAPI Spec (`openapi.json`)**: Exported and ready for Postman import.
