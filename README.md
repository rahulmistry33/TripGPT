# ✈️ TripGPT: Multi-Agent Travel Planner (`trip-gpt`)

`trip-gpt` is a production-grade multi-agent travel planning system built with **LangGraph**, **FastAPI**, and 100% **Model Context Protocol (MCP)** tools (AviationStack flight tools, Tavily hotel & search tools, and a custom stdio Weather MCP server).

---

## 🚀 Quick Start with `uv`

`uv` is the recommended fast Python runner for this repository.

### 1. Install Dependencies
```bash
uv sync
```

### 2. Configure Environment Variables (`.env`)
Ensure `.env` in the root directory contains your API keys:
```env
GROQ_API_KEY=your_groq_api_key
TAVILY_API_KEY=your_tavily_api_key
AVIATIONSTACK_API_KEY=your_aviationstack_api_key
```

---

## 🌐 Running the FastAPI Web Server with `uv`

To launch the FastAPI server using `uv`:

```bash
uv run python run_server.py
```

Or run Uvicorn directly with `uv`:

```bash
uv run uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

### 📖 Interactive API Documentation (Swagger)
Once the server is running, open your browser and navigate to:
👉 **[http://localhost:8000/docs](http://localhost:8000/docs)**

---

## 💻 Running the CLI Interactive Session

To run the multi-turn CLI runner:

```bash
uv run python main.py
```

---

## 🧪 Running Automated Tests & Verification with `uv`

- **FastAPI Endpoints Verification**:
  ```bash
  uv run python test_fastapi_app.py
  ```

- **MCP Multi-Server Verification**:
  ```bash
  uv run python test_mcp_servers.py
  ```

---

## 📡 API Endpoints Overview

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/` | API status and documentation link |
| `POST` | `/api/v1/chat` | Send a user message turn & execute multi-agent planning |
| `GET` | `/api/v1/users/{user_id}/threads` | List all trip conversation threads for a user |
| `POST` | `/api/v1/users/{user_id}/threads` | Create a new trip thread session |
| `GET` | `/api/v1/users/{user_id}/threads/{thread_id}` | Retrieve thread message history & extracted trip details |
| `DELETE` | `/api/v1/users/{user_id}/threads/{thread_id}` | Delete a trip thread session |
