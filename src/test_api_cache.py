from unittest.mock import Mock, patch

from fastapi.testclient import TestClient

from src.api import app
from src.cache import build_query_cache_key, delete_cache


client = TestClient(app)


def test_query_cache_miss_then_hit():
    fake_agent = Mock()

    fake_agent.invoke.return_value = {
        "answer": "UU 1 Tahun 2024 mengatur tentang ...",
        "sources": [],
        "citations": [],
        "plan": ["retrieve"],
        "completed_steps": ["retrieve"],
        "missing_regulations": [],
        "comparison": None,
        "sql_results": [],
        "external_sources": [],
        "external_citations": [],
    }

    question = "Apa isi UU 1 Tahun 2024?"

    cache_key = build_query_cache_key(question)
    delete_cache("query", cache_key)
    try:
        with patch("src.api.get_agent", return_value=fake_agent):
            response_1 = client.post(
                "/query",
                json={"question": question},
            )

            response_2 = client.post(
                "/query",
                json={"question": question},
            )
    finally:
        delete_cache("query", cache_key)

    assert response_1.status_code == 200
    assert response_2.status_code == 200

    assert response_1.json() == response_2.json()

    # Agent should only execute once.
    fake_agent.invoke.assert_called_once_with(question)
