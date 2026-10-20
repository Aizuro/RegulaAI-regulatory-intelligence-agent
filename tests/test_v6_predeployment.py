"""V6 live predeployment smoke tests.

These tests call a running API at BASE_URL and may invoke real dependencies
(Redis, retrieval, and external research). They are opt-in so a normal unit-test
run does not fail merely because the deployment server is not running.

To run them intentionally:
    PowerShell:
        $env:RUN_V6_PREDEPLOYMENT = "1"
        pytest tests/test_v6_predeployment.py -v
"""
import os
import unittest
import requests


BASE_URL = os.getenv("REGULA_API_BASE_URL", "http://127.0.0.1:8000")
RUN_LIVE_TESTS = os.getenv("RUN_V6_PREDEPLOYMENT") == "1"


@unittest.skipUnless(
    RUN_LIVE_TESTS,
    "Live predeployment tests are opt-in; set RUN_V6_PREDEPLOYMENT=1 after starting the API.",
)
class TestV6PreDeployment(unittest.TestCase):

    def query(self, question):
        response = requests.post(
            f"{BASE_URL}/query",
            json={"question": question},
            timeout=60,
        )

        self.assertEqual(
            response.status_code,
            200,
            msg=f"POST /query gagal: {response.status_code}\n{response.text}",
        )

        return response.json()

    def test_health(self):
        response = requests.get(f"{BASE_URL}/health", timeout=10)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data.get("status"), "ok")
        self.assertEqual(
            data.get("service"),
            "regulatory-intelligence-agent",
        )

    def test_internal_regulation_query(self):
        data = self.query("Apa ketentuan UU Nomor 1 Tahun 2024?")
        self.assertTrue(data.get("answer"))
        self.assertIn("sources", data)
        self.assertIsInstance(data["sources"], list)
        for source in data["sources"]:
            self.assertIsInstance(
                source, dict, msg=f"Source bukan dictionary: {type(source)}"
            )

    def test_regulatory_comparison(self):
        data = self.query(
            "Bandingkan UU Nomor 1 Tahun 2024 dengan UU Nomor 3 Tahun 2024"
        )
        self.assertTrue(data.get("answer"))
        self.assertIn("comparison", data)
        self.assertIsNotNone(data["comparison"])
        self.assertIsInstance(data["comparison"], dict)
        self.assertIn("compare_regulations", data.get("plan", []))
        self.assertIsInstance(data.get("sources"), list)
        for source in data["sources"]:
            self.assertIsInstance(
                source,
                dict,
                msg=f"Comparison menghasilkan source yang bukan dictionary: {type(source)}",
            )

    def test_sql_analysis(self):
        data = self.query("Berapa jumlah regulasi berdasarkan sektor?")
        self.assertTrue(data.get("answer"))
        self.assertIn("sql_results", data)
        self.assertIsInstance(data["sql_results"], list)
        self.assertGreater(len(data["sql_results"]), 0, msg="SQL results kosong.")

    def test_missing_regulation(self):
        data = self.query(
            "Apa ketentuan Peraturan Presiden Nomor 999 Tahun 2099?"
        )
        self.assertIn("missing_regulations", data)
        self.assertIsInstance(data["missing_regulations"], list)

    def test_external_research_contract(self):
        data = self.query(
            "Apa ketentuan Peraturan Presiden Nomor 999 Tahun 2099?"
        )
        self.assertIn("external_sources", data)
        self.assertIn("external_citations", data)
        self.assertIsInstance(data["external_sources"], list)
        self.assertIsInstance(data["external_citations"], list)

    def test_external_citations_are_valid(self):
        data = self.query(
            "Apa ketentuan Peraturan Presiden Nomor 999 Tahun 2099?"
        )
        for citation in data.get("external_citations", []):
            self.assertIsInstance(citation, dict)
            url = citation.get("url", "")
            self.assertTrue(
                url.startswith("https://"),
                msg=f"External citation URL tidak HTTPS: {url}",
            )
            self.assertTrue(
                citation.get("title"),
                msg="External citation tidak memiliki title.",
            )

    def test_response_contract(self):
        data = self.query("Apa ketentuan UU Nomor 1 Tahun 2024?")
        required_fields = [
            "answer",
            "sources",
            "citations",
            "plan",
            "completed_steps",
            "missing_regulations",
            "comparison",
            "sql_results",
            "external_sources",
            "external_citations",
        ]
        for field in required_fields:
            self.assertIn(field, data, msg=f"Field response hilang: {field}")

    def test_empty_question(self):
        response = requests.post(
            f"{BASE_URL}/query",
            json={"question": "   "},
            timeout=10,
        )
        self.assertEqual(response.status_code, 400)

    def test_missing_question(self):
        response = requests.post(
            f"{BASE_URL}/query",
            json={},
            timeout=10,
        )
        self.assertEqual(response.status_code, 422)


if __name__ == "__main__":
    unittest.main(verbosity=2)
