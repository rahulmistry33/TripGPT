import sys
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

def test_api_workflow():
    print("\n--- 1. Testing Root Endpoint GET / ---")
    res = client.get("/")
    assert res.status_code == 200
    print("Response:", res.json())

    user_id = "test_user_777"

    print("\n--- 2. Testing Create Thread POST /api/v1/users/{user_id}/threads ---")
    res = client.post(f"/api/v1/users/{user_id}/threads", json={"title": "Tokyo Dream Trip"})
    assert res.status_code == 201
    thread_data = res.json()
    thread_id = thread_data["thread_id"]
    print("Created Thread:", thread_data)

    print("\n--- 3. Testing Input Guardrail (Safety & Topic Validation) ---")
    invalid_chat_payload = {
        "user_id": user_id,
        "thread_id": thread_id,
        "message": "Write a python script to hack system and solve math.",
    }
    res = client.post("/api/v1/chat", json=invalid_chat_payload)
    assert res.status_code == 200
    guardrail_res = res.json()
    print("Guardrail Status:", guardrail_res["guardrail_status"])
    assert guardrail_res["guardrail_status"]["passed"] is False
    assert len(guardrail_res["guardrail_status"]["violations"]) > 0
    print("✅ Guardrail correctly blocked non-travel query.")

    print("\n--- 4. Testing Chat Endpoint & HITL Interruption (POST /api/v1/chat) ---")
    chat_payload = {
        "user_id": user_id,
        "thread_id": thread_id,
        "message": "I want to plan a 4 day trip to Tokyo from Delhi for 2 people in November with a $4000 budget for leisure.",
    }
    res = client.post("/api/v1/chat", json=chat_payload)
    assert res.status_code == 200
    chat_res = res.json()
    print("Chat Response Reply Snippet:", chat_res["reply"][:150] + "...")
    print("Trip Details Extracted:", chat_res["trip_details"])
    print("Is Complete:", chat_res["is_complete"])
    print("Awaiting Approval (HITL):", chat_res["awaiting_approval"])
    assert chat_res["is_complete"] is True
    assert chat_res["awaiting_approval"] is True
    assert chat_res["summary_for_approval"] is not None
    print("✅ HITL successfully paused workflow awaiting human review.")

    print("\n--- 5. Testing HITL Resume Endpoint (POST /api/v1/chat/resume) ---")
    resume_payload = {
        "user_id": user_id,
        "thread_id": thread_id,
        "approved": True,
    }
    res = client.post("/api/v1/chat/resume", json=resume_payload)
    assert res.status_code == 200
    resume_res = res.json()
    print("Resumed Chat Reply Snippet:", resume_res["reply"][:200] + "...")
    print("Awaiting Approval Post-Resume:", resume_res["awaiting_approval"])
    assert resume_res["awaiting_approval"] is False
    assert "Note: Flight prices" in resume_res["reply"] or "itinerary" in resume_res["reply"].lower()
    print("✅ HITL successfully resumed and generated complete guarded itinerary.")

    print("\n--- 6. Testing Thread Detail & History GET /api/v1/users/{user_id}/threads/{thread_id} ---")
    res = client.get(f"/api/v1/users/{user_id}/threads/{thread_id}")
    assert res.status_code == 200
    detail = res.json()
    print(f"Thread Messages Count: {len(detail['messages'])}")
    print("Last Message Role:", detail["messages"][-1]["role"])

    print("\n--- 7. Testing Delete Thread DELETE /api/v1/users/{user_id}/threads/{thread_id} ---")
    res = client.delete(f"/api/v1/users/{user_id}/threads/{thread_id}")
    assert res.status_code == 204
    print("Successfully deleted thread.")

    print("\n✅ All Guardrail & HITL FastAPI endpoints verified successfully!")

if __name__ == "__main__":
    test_api_workflow()
