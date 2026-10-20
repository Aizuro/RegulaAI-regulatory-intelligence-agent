"""Evaluation suite for the V5 FastAPI layer.

These tests mock the V4 agent so they focus on API behavior rather than
repeating the V4 integration tests.
"""

from pathlib import Path
from typing import Any, Dict
from unittest.mock import patch

from fastapi.testclient import TestClient
from src import api


class FakeAgent:
    """Minimal V4-compatible agent for API contract tests."""

    def __init__(self, result: Dict[str, Any]):
        self.result = result
        self.questions = []

    def invoke(self, question: str) -> Dict[str, Any]:
        self.questions.append(question)
        return self.result


def build_fake_agent() -> FakeAgent:
    return FakeAgent(
        {
            "answer": "Contoh jawaban dari Regulatory Intelligence Agent.",
            "sources": [
                {
                    "page_content": "Contoh evidence.",
                    "metadata": {"source": "2024uu001.pdf"},
                }
            ],
            "citations": [
                {
                    "source": "2024uu001.pdf",
                    "document_type": "UU",
                    "regulation_number": "1",
                    "year": "2024",
                    "pasal": "17",
                    "ayat": "2a",
                    "page_reference": "Halaman 10",
                }
            ],
            "plan": ["get_regulation"],
            "completed_steps": ["get_regulation"],
            "missing_regulations": [],
            "comparison": None,
            "sql_results": [],
        }
    )


def run_test(name: str, function) -> bool:
    try:
        function()
        print(f"{name:<28}: PASS")
        return True
    except Exception as exc:
        print(f"{name:<28}: FAIL")
        print(f"  {type(exc).__name__}: {exc}")
        return False


def test_health() -> None:
    client = TestClient(api.app)
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["version"] == "7.1.0"


def test_query_contract() -> None:
    fake_agent = build_fake_agent()
    api._agent = fake_agent

    client = TestClient(api.app)
    with patch("src.api.get_cache", return_value=None), patch("src.api.set_cache"):
        response = client.post(
            "/query",
            json={"question": "Apa ketentuan UU Nomor 1 Tahun 2024?"},
        )

    assert response.status_code == 200

    body = response.json()

    assert body["answer"]
    assert body["plan"] == ["get_regulation"]
    assert body["completed_steps"] == ["get_regulation"]
    assert body["citations"][0]["source"] == "2024uu001.pdf"
    assert fake_agent.questions == ["Apa ketentuan UU Nomor 1 Tahun 2024?"]


def test_question_is_trimmed() -> None:
    fake_agent = build_fake_agent()
    api._agent = fake_agent

    client = TestClient(api.app)
    with patch("src.api.get_cache", return_value=None), patch("src.api.set_cache"):
        response = client.post(
            "/query",
            json={"question": "   test question   "},
        )

    assert response.status_code == 200
    assert fake_agent.questions == ["test question"]


def test_empty_question_rejected() -> None:
    client = TestClient(api.app)
    response = client.post("/query", json={"question": "   "})

    # Whitespace-only input passes Pydantic's min_length validation,
    # then is rejected by the endpoint after stripping whitespace.
    assert response.status_code == 400
    assert response.json()["detail"] == "question tidak boleh kosong."


def test_missing_question_rejected() -> None:
    client = TestClient(api.app)
    response = client.post("/query", json={})

    assert response.status_code == 422


def test_agent_error_returns_500() -> None:
    class ErrorAgent:
        def invoke(self, question: str):
            raise RuntimeError("synthetic failure")

    api._agent = ErrorAgent()

    client = TestClient(api.app)
    with patch("src.api.get_cache", return_value=None), patch("src.api.set_cache"):
        response = client.post("/query", json={"question": "test"})

    assert response.status_code == 500
    assert "Agent execution failed" in response.json()["detail"]


def main() -> int:
    print("=" * 78)
    print("REGULATORY INTELLIGENCE AGENT V5 — API EVALUATION")
    print("=" * 78)

    tests = [
        ("Health endpoint", test_health),
        ("Query contract", test_query_contract),
        ("Question trimming", test_question_is_trimmed),
        ("Empty question", test_empty_question_rejected),
        ("Missing question", test_missing_question_rejected),
        ("Agent error handling", test_agent_error_returns_500),
    ]

    passed = sum(run_test(name, test) for name, test in tests)
    total = len(tests)

    print()
    print("=" * 78)
    print("V5 EVALUATION SUMMARY")
    print("=" * 78)
    print(f"Tests passed            : {passed}/{total}")
    print(f"Tests failed            : {total - passed}/{total}")
    print(f"Overall status          : {'PASS' if passed == total else 'FAIL'}")
    print("=" * 78)

    return 0 if passed == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
