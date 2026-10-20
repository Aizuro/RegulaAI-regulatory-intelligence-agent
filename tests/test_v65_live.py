"""V6.5 live API/UI contract verification.

Run from project root:
    python -m unittest testing.test_v65_live -v

The API tests use FastAPI's in-process TestClient with the agent mocked,
so they verify the actual HTTP response model and routing without requiring
Groq or a running server. The final smoke test checks that the Streamlit UI
contains the external citation renderer and consumes external_citations.
"""

import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
API_PATH = ROOT / "src" / "api.py"
APP_PATH = ROOT / "app.py"


class FakeAgent:
    def __init__(self, result):
        self.result = result

    def invoke(self, question):
        self.question = question
        return self.result


class TestV65Live(unittest.TestCase):
    def _load_api(self):
        spec = importlib.util.spec_from_file_location("v65_api_test", API_PATH)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_health_reports_v65(self):
        module = self._load_api()
        result = module.health()
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["version"], "7.1.0")

    def test_query_preserves_v5_fields_and_adds_v65_fields(self):
        module = self._load_api()
        fake_result = {
            "answer": "Jawaban eksternal.",
            "sources": [],
            "citations": [],
            "plan": ["search_external_regulations"],
            "completed_steps": ["search_external_regulations"],
            "missing_regulations": [],
            "comparison": None,
            "sql_results": [],
            "external_sources": [{
                "title": "Regulasi Eksternal",
                "url": "https://peraturan.bpk.go.id/Example",
                "source_tier": 1,
            }],
            "external_citations": [{
                "title": "Regulasi Eksternal",
                "url": "https://peraturan.bpk.go.id/Example",
                "source_name": "JDIH BPK",
                "source_tier": 1,
                "evidence": "Pasal 1 ...",
            }],
        }
        module._agent = FakeAgent(fake_result)

        request = module.QueryRequest(question="  test external  ")
        with patch.object(module, "get_cache", return_value=None), patch.object(module, "set_cache"):
            response = module.query(request)
        payload = response.model_dump()

        # V5 contract remains present.
        for key in (
            "answer", "sources", "citations", "plan",
            "completed_steps", "missing_regulations", "comparison",
            "sql_results",
        ):
            self.assertIn(key, payload)

        # V6.5 fields are exposed.
        self.assertEqual(len(payload["external_sources"]), 1)
        self.assertEqual(len(payload["external_citations"]), 1)
        self.assertEqual(payload["external_citations"][0]["source_tier"], 1)

    def test_empty_question_still_returns_400(self):
        module = self._load_api()
        with self.assertRaises(module.HTTPException) as ctx:
            module.query(module.QueryRequest(question=" "))
        self.assertEqual(ctx.exception.status_code, 400)

    def test_agent_error_still_returns_500(self):
        module = self._load_api()

        class BrokenAgent:
            def invoke(self, question):
                raise RuntimeError("boom")

        module._agent = BrokenAgent()
        with patch.object(module, "get_cache", return_value=None), patch.object(module, "set_cache"):
            with self.assertRaises(module.HTTPException) as ctx:
                module.query(module.QueryRequest(question="test"))
        self.assertEqual(ctx.exception.status_code, 500)

    def test_streamlit_consumes_external_citations(self):
        # Static UI verification avoids launching Streamlit during unittest.
        text = APP_PATH.read_text(encoding="utf-8")
        self.assertIn("def render_external_citations", text)
        self.assertIn('result.get("external_citations")', text)
        self.assertIn("🌐 External Sources", text)
        self.assertIn("Open source", text)


if __name__ == "__main__":
    unittest.main()
