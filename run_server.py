import sys
import uvicorn
from dotenv import load_dotenv

# Ensure UTF-8 output encoding for Windows terminal
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Load environment variables (.env)
load_dotenv()

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 8000))
    print(f"Starting TripGPT API Server on http://0.0.0.0:{port} ...")
    print(f"Interactive Swagger Documentation available at http://localhost:{port}/docs")
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=port,
        reload=False,
    )
